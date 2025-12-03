from datetime import datetime

import torch
from typing import Literal, Optional, Callable, Dict, List

import wandb
from torch.optim import AdamW
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
from .initializer import VLLMInitializer
from .gen_prompt import PromptDataset
from .sft_helper import tokenize_prompt_and_output, get_response_log_probs
from .grpo_helper import compute_group_normalized_rewards, grpo_microbatch_train_step
from .drgrpo_grader import r1_zero_reward_fn

class Grpo:
    def __init__(
        self,
        policy: PreTrainedModel,  # 核心策略模型（nn.Module/Transformers PreTrainedModel）
        old_policy_wrapper: VLLMInitializer,    # inference only
        reward_fn: Callable[[str, str], Dict[str, float]], # (response, ground_truth) -> reward
        raw_questions: List[str], 
        raw_ground_truths: List[str],  
        tokenizer: PreTrainedTokenizer,  # 文本tokenizer

        # 算法核心参数
        n_grpo_steps: int = 200,
        learning_rate: float = 1e-5,
        advantage_eps: float = 1e-6,
        rollout_batch_size: int = 256,
        group_size: int = 8,
        sampling_temperature: float = 1.0,
        sampling_min_tokens: int = 4,
        sampling_max_tokens: int = 1024,
        epochs_per_rollout_batch: int = 1,
        train_batch_size: int = 256,
        gradient_accumulation_steps: int = 128,
        gpu_memory_utilization: float = 0.85,
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"] = "reinforce_with_baseline",
        use_std_normalization: bool = True,
    ):
        # assert
        assert train_batch_size % gradient_accumulation_steps == 0, "train_batch_size must be divisible by gradient_accumulation_steps"
        self.micro_train_batch_size = train_batch_size // gradient_accumulation_steps

        assert rollout_batch_size % group_size == 0, "rollout_batch_size must be divisible by group_size"
        self.n_prompts_per_rollout_batch = rollout_batch_size // group_size

        assert train_batch_size >= group_size, "train_batch_size must be greater than or equal to group_size"
        self.n_microbatches_per_rollout_batch = rollout_batch_size // self.micro_train_batch_size

        self.policy = policy
        self.old_policy_wrapper = old_policy_wrapper
        self.old_policy = old_policy_wrapper.inf_vllm
        self.reward_fn = reward_fn
        self.tokenizer = tokenizer

        self.n_grpo_steps = n_grpo_steps
        self.learning_rate = learning_rate
        self.advantage_eps = advantage_eps
        self.group_size = group_size
        self.epochs_per_rollout_batch = epochs_per_rollout_batch
        self.loss_type = loss_type
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.use_std_normalization = use_std_normalization

        self.sampling_params = SamplingParams(
            temperature=sampling_temperature,
            min_tokens=sampling_min_tokens,
            max_tokens=sampling_max_tokens,
            n=group_size,
            stop=["</answer>"],
            top_p=1.0,
        )
        self.prompt_dataset = PromptDataset(
            raw_question=raw_questions,
            raw_ground_truths=raw_ground_truths,
        )
        self.optimizer = AdamW(
            policy.parameters(),
            lr=learning_rate,
            weight_decay=0.0,
            betas=(0.9, 0.95),
        )

    def train(self):
        for step in range(self.n_grpo_steps):
            # Sample a batch of questions D_b from D
            prompts, ground_truths = self.prompt_dataset.sample_batch(self.n_prompts_per_rollout_batch)
            
            # Set the old policy model πθold ←πθ
            self.old_policy_wrapper.load_policy_into_vllm(self.policy)
            
            # Sample G outputs for each question q ∈ D_b
            outputs = self.old_policy.generate(prompts, self.sampling_params)

            repeated_prompts: List[str] = []
            responses: List[str] = []
            old_log_probs: List[List[float]] = []   # ToDo: reinforce_with_baseline do not need
            for completion in outputs:
                repeated_prompts.extend([completion.prompt for _ in completion.outputs])
                responses.extend([res.text for res in completion.outputs])
                old_log_probs.extend([res.log_probs for res in completion.outputs])
                
            # Compute rewards and advantages for each sampled output o(i) by running reward function R(q, o(i))
            advantages, raw_rewards, metadata = compute_group_normalized_rewards(
                reward_fn=self.reward_fn,
                rollout_responses=responses,
                repeated_ground_truths=[truth for truth in ground_truths for _ in range(self.group_size)],
                group_size=self.group_size,
                advantage_eps=self.advantage_eps,
                normalize_by_std=self.use_std_normalization,
            )
            advantages, raw_rewards = advantages.unsqueeze(1), raw_rewards.unsqueeze(1)

            # tokenizer inputs
            combo = tokenize_prompt_and_output(
                prompt_strs=repeated_prompts,
                output_strs=responses,
                tokenizer=self.tokenizer,
            )
            input_rollouts, labels, response_mask = (
                combo["input_ids"], combo["labels"], combo["response_mask"])
            
            for epoch in range(self.epochs_per_rollout_batch):
                for micro_step in range(self.n_microbatches_per_rollout_batch):
                    start_idx = micro_step * self.micro_train_batch_size
                    end_idx = start_idx + self.micro_train_batch_size

                    micro_x = input_rollouts[start_idx:end_idx]
                    micro_y = labels[start_idx:end_idx]
                    micro_mask = response_mask[start_idx:end_idx]
                    micro_rewards = raw_rewards[start_idx:end_idx]
                    micro_advantages = advantages[start_idx:end_idx]

                    policy_log_probs = get_response_log_probs(
                        model=self.policy,
                        input_ids=micro_x,
                        labels=micro_y,
                    )["log_probs"]

                    loss, metadata = grpo_microbatch_train_step(
                        policy_log_probs=policy_log_probs,
                        response_mask=micro_mask,
                        gradient_accumulation_steps=self.gradient_accumulation_steps,
                        loss_type=self.loss_type,
                        raw_rewards=micro_rewards,
                        advantages=micro_advantages,
                        # ToDo: reinforce_with_baseline do not need
                        # old_log_probs=old_log_probs,
                    )

                    # do step for each gradient_accumulation_steps
                    if (micro_step + 1) % self.gradient_accumulation_steps == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()


if __name__ == "__main__":
    wandb.init(
        project="cs336_assignment5",
        name="grpo" + datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )

    # init policy
    model_path = "./data/models/Qwen2.5-Math-1.5B"
    policy = AutoModelForCausalLM.from_pretrained(
        model_path,
        # torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    wandb.watch(policy, log="all")

    # init vllm
    inf_vllm = VLLMInitializer(
        model_id=model_path,
        device="cpu",   # ToDo: compensate
        seed=666,
    )

    # ToDo: gen questions and corresponding truths


    # grpo train
    grpo = Grpo(
        policy=policy,
        old_policy_wrapper=inf_vllm,
        reward_fn=r1_zero_reward_fn,
        raw_questions=["What is the capital of China?"],
        raw_ground_truths=["What is the capital of China?"],
        tokenizer=tokenizer,
    )
    grpo.train()

    wandb.finish()