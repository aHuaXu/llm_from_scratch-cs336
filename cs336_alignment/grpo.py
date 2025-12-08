from datetime import datetime
from logging import getLogger

import torch
from typing import Literal, Optional, Callable, Dict, List

import wandb
from torch.optim import AdamW
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
from .vllm_wrapper import VLLMWrapper
from .gen_prompt import PromptDataset
from .sft_helper import tokenize_prompt_and_output, get_response_log_probs, get_model
from .grpo_helper import compute_group_normalized_rewards, grpo_microbatch_train_step
from .drgrpo_grader import r1_zero_reward_fn
from .init import log_init

logger = getLogger(__name__)

class Grpo:
    def __init__(
        self,
        policy: PreTrainedModel,  # 核心策略模型（nn.Module/Transformers PreTrainedModel）
        old_policy_wrapper: VLLMWrapper,    # inference only
        tokenizer: PreTrainedTokenizer,  # 文本tokenizer
        reward_fn: Callable[[str, str], Dict[str, float]] = r1_zero_reward_fn,  # (response, ground_truth) -> reward

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
        self.prompt_dataset = PromptDataset()
        self.optimizer = AdamW(
            policy.parameters(),
            lr=learning_rate,
            weight_decay=0.0,
            betas=(0.9, 0.95),
        )

        wandb.config.update({
            "n_grpo_steps": n_grpo_steps,
            "learning_rate": learning_rate,
            "advantage_eps": advantage_eps,
            "rollout_batch_size": rollout_batch_size,
            "group_size": group_size,
            "sampling_temperature": sampling_temperature,
            "sampling_min_tokens": sampling_min_tokens,
            "sampling_max_tokens": sampling_max_tokens,
            "epochs_per_rollout_batch": epochs_per_rollout_batch,
            "train_batch_size": train_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "loss_type": loss_type,
            "use_std_normalization": use_std_normalization,
            "micro_train_batch_size": self.micro_train_batch_size,
            "n_prompts_per_rollout_batch": self.n_prompts_per_rollout_batch,
        })

    def train(self):
        for step in range(self.n_grpo_steps):
            current_step = step + 1
            step_start = datetime.now()
            logger.info(f"\n=== Starting GRPO step {current_step}/{self.n_grpo_steps} ===")

            # Sample a batch of questions D_b from D
            prompts, _, ground_truths = self.prompt_dataset.sample_batch(self.n_prompts_per_rollout_batch)
            
            # Set the old policy model πθold ←πθ
            self.old_policy_wrapper.load_policy_into_vllm(self.policy)
            
            # Sample G outputs for each question q ∈ D_b
            gen_start = datetime.now()
            outputs = self.old_policy_wrapper.generate(prompts, self.sampling_params)
            gen_duration = (datetime.now() - gen_start).total_seconds()

            repeated_prompts: List[str] = []
            responses: List[str] = []
            old_log_probs: List[List[float]] = []  # ToDo: reinforce_with_baseline do not need
            for completion in outputs:
                repeated_prompts.extend([completion.prompt for _ in completion.outputs])
                responses.extend([res.text for res in completion.outputs])
                old_log_probs.extend([res.log_probs for res in completion.outputs])

            logger.info(
                f"Step {current_step}: Generated {len(responses)} responses ({self.group_size} per prompt) in {gen_duration:.2f}s")
            wandb.log({
                "train_step": current_step,
                "train/response_count": len(responses),
                "train/generation_duration": gen_duration,
                "train/responses_per_prompt": self.group_size
            })

            # Compute rewards and advantages for each sampled output o(i) by running reward function R(q, o(i))
            reward_start = datetime.now()
            advantages, raw_rewards, metadata = compute_group_normalized_rewards(
                reward_fn=self.reward_fn,
                rollout_responses=responses,
                repeated_ground_truths=[truth for truth in ground_truths for _ in range(self.group_size)],
                group_size=self.group_size,
                advantage_eps=self.advantage_eps,
                normalize_by_std=self.use_std_normalization,
            )
            advantages, raw_rewards = advantages.unsqueeze(1), raw_rewards.unsqueeze(1)
            reward_duration = (datetime.now() - reward_start).total_seconds()

            avg_reward = raw_rewards.mean().item()
            avg_advantage = advantages.mean().item()
            logger.info(
                f"Step {current_step}: Computed rewards/advantages in {reward_duration:.2f}s | Avg reward: {avg_reward:.4f} | Avg advantage: {avg_advantage:.4f}")
            wandb.log({
                "train_step": current_step,
                "train/reward_computation_duration": reward_duration,
                "train/avg_reward": raw_rewards.mean().item(),
                "train/avg_advantage": advantages.mean().item()
            })

            # tokenizer inputs
            token_start = datetime.now()
            combo = tokenize_prompt_and_output(
                prompt_strs=repeated_prompts,
                output_strs=responses,
                tokenizer=self.tokenizer,
            )
            input_rollouts, labels, response_mask = (
                combo["input_ids"], combo["labels"], combo["response_mask"])
            token_duration = (datetime.now() - token_start).total_seconds()
            logger.info(
                f"Step {current_step}: Tokenized {len(repeated_prompts)} prompt-response pairs in {token_duration:.2f}s | Sequence length: {input_rollouts.shape[1]}")
            wandb.log({
                "train_step": current_step,
                "train/tokenization_duration": token_duration,
                "train/sequence_length": input_rollouts.shape[1]
            })

            epoch_losses = []
            for epoch in range(self.epochs_per_rollout_batch):
                epoch_start = datetime.now()
                logger.info(f"Step {current_step}: Starting training epoch {epoch + 1}/{self.epochs_per_rollout_batch}")

                micro_step_losses = []
                for micro_step in range(self.n_microbatches_per_rollout_batch):
                    micro_start = datetime.now()
                    current_micro_step = micro_step + 1

                    start_idx = micro_step * self.micro_train_batch_size
                    end_idx = start_idx + self.micro_train_batch_size

                    micro_x = input_rollouts[start_idx:end_idx]
                    micro_y = labels[start_idx:end_idx]
                    micro_mask = response_mask[start_idx:end_idx]
                    micro_rewards = raw_rewards[start_idx:end_idx]
                    micro_advantages = advantages[start_idx:end_idx]

                    logger.debug(
                        f"Step {current_step}, Epoch {epoch + 1}, Micro-step {current_micro_step}: Processing batch {start_idx}-{end_idx} (size: {self.micro_train_batch_size})")

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

                    loss_value = loss.item()
                    micro_step_losses.append(loss_value)
                    micro_duration = (datetime.now() - micro_start).total_seconds()
                    logger.debug(
                        f"Step {current_step}, Epoch {epoch + 1}, Micro-step {current_micro_step}: Loss = {loss_value:.4f} | Duration = {micro_duration:.2f}s")
                    wandb.log({
                        "grpo_step": current_step,
                        "training/micro_step_loss": loss_value,
                        "training/micro_step_duration": micro_duration,
                        "training/epoch": epoch + 1,
                        "training/micro_step": current_micro_step
                    })

                    # do step for each gradient_accumulation_steps
                    if (micro_step + 1) % self.gradient_accumulation_steps == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        logger.debug(
                            f"Step {current_step}, Epoch {epoch + 1}: Optimizer step completed (gradient accumulation window finished)")

                avg_epoch_loss = sum(micro_step_losses) / len(micro_step_losses) if micro_step_losses else 0.0
                epoch_losses.append(avg_epoch_loss)
                epoch_duration = (datetime.now() - epoch_start).total_seconds()
                logger.info(
                    f"Step {current_step}: Epoch {epoch + 1} completed | Avg loss: {avg_epoch_loss:.4f} | Duration: {epoch_duration:.2f}s")
                wandb.log({
                    "grpo_step": current_step,
                    "training/epoch_loss": avg_epoch_loss,
                    "training/epoch_duration": epoch_duration,
                    "training/epoch_number": epoch + 1
                })

            step_duration = (datetime.now() - step_start).total_seconds()
            avg_step_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
            logger.info(f"\n=== Completed GRPO Step {current_step}/{self.n_grpo_steps} ==="
                        f"\n  Step duration: {step_duration:.2f}s"
                        f"\n  Avg epoch loss: {avg_step_loss:.4f}"
                        f"\n  Learning rate: {self.optimizer.param_groups[0]['lr']:.6f}")

            wandb.log({
                "grpo_step": current_step,
                "grpo/step_duration": step_duration,
                "grpo/avg_step_loss": avg_step_loss,
                "grpo/learning_rate": self.optimizer.param_groups[0]['lr'],
                "grpo/group_size": self.group_size,
                "grpo/loss_type": self.loss_type
            })

        total_train_duration = (datetime.now() - self.train_start_time).total_seconds()
        logger.info(f"\n=== GRPO Training Complete ==="
                    f"\n  Total steps: {self.n_grpo_steps}"
                    f"\n  Total duration: {total_train_duration:.2f}s ({total_train_duration / 60:.2f} mins)"
                    f"\n  Final learning rate: {self.optimizer.param_groups[0]['lr']:.6f}")


if __name__ == "__main__":
    log_init(task_name="grpo")

    # init policy
    model_path = "./data/models/Qwen2.5-Math-1.5B"
    policy, tokenizer, inf_vllm = get_model(model_path)
    wandb.watch(policy, log="all")
    logger.info("load model successfully")

    # grpo train
    grpo = Grpo(
        policy=policy,
        old_policy_wrapper=inf_vllm,
        tokenizer=tokenizer,
    )
    grpo.train()

    wandb.finish()