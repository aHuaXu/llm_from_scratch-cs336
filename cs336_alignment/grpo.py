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
from .sft_helper import tokenize_prompt_and_output, get_response_log_probs, get_model, save_policy
from .grpo_helper import compute_group_normalized_rewards, grpo_microbatch_train_step
from .drgrpo_grader import r1_zero_reward_fn
from .init import log_init, train_device, env_init
from .evaluate import evaluate_vllm

env_init()
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
        epochs_per_rollout_batch: int = 1,  # 只考虑1了，每轮数据只训练一次
        train_batch_size: int = 256,
        gradient_accumulation_steps: int = 128,
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"] = "reinforce_with_baseline",
        use_std_normalization: bool = True,
        validate_interval: int = 10,
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
        self.validate_interval = validate_interval

        self.eval_step_counter = 0

        self.sampling_params = SamplingParams(
            temperature=sampling_temperature,
            min_tokens=sampling_min_tokens,
            max_tokens=sampling_max_tokens,
            n=group_size,
            stop=["</answer>"],
            top_p=1.0,
            include_stop_str_in_output=True,
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

    def evaluate(self, current_train_step: int):
        eval_start = datetime.now()
        self.eval_step_counter += 1

        logger.info(
            f"Step {current_train_step}: Starting evaluation (eval step {self.eval_step_counter})...")

        # Load current policy into inference model
        self.old_policy_wrapper.load_policy_into_vllm(self.policy)
        logger.debug(
            f"Step {current_train_step}: Loaded policy model into VLLM inference wrapper")

        # Run evaluation
        format_accuracy, accuracy = evaluate_vllm(self.old_policy_wrapper)
        eval_duration = (datetime.now() - eval_start).total_seconds()

        # Log evaluation results
        logger.info(
            f"Evaluation {self.eval_step_counter} complete | "
            f"Format accuracy: {format_accuracy:.4f} | "
            f"Accuracy: {accuracy:.4f} | "
            f"Evaluation duration: {eval_duration:.2f}s"
        )

        # Log to wandb (removed run check)
        wandb.log({
            "eval_step": self.eval_step_counter,
            "eval/format_accuracy": format_accuracy,
            "eval/accuracy": accuracy,
            "eval/duration": eval_duration,
            "eval/corresponding_train_step": current_train_step
        })

        # save checkpoint
        save_policy(self.policy, self.tokenizer, "./data/models/Qwen2.5-Math-1.5B-grpo")


    def train(self):
        train_start_time = datetime.now()
        self.eval_step_counter = 0
        for step in range(self.n_grpo_steps):
            current_step = step + 1
            step_start = datetime.now()
            logger.info(f"\n=== Starting GRPO step {current_step}/{self.n_grpo_steps} ===")

            # 1. Sample a batch of questions D_b from D
            prompts, _, ground_truths = self.prompt_dataset.train_batch(self.n_prompts_per_rollout_batch)
            
            # 2. Sample G outputs for each question q ∈ D_b
            gen_start = datetime.now()
            outputs = self.old_policy_wrapper.generate(prompts, self.sampling_params)
            gen_duration = (datetime.now() - gen_start).total_seconds()

            repeated_prompts: List[str] = []
            responses: List[str] = []
            for completion in outputs:
                repeated_prompts.extend([completion.prompt for _ in completion.outputs])
                responses.extend([res.text for res in completion.outputs])

            logger.info(
                f"Step {current_step}: Generated {len(responses)} "
                f"responses in {gen_duration:.2f}s")
            wandb.log({
                "train_step": current_step,
                "train/response_count": len(responses),
                "train/generation_duration": gen_duration,
                "train/responses_per_prompt": self.group_size
            })

            # 3. Compute rewards and advantages for each sampled output o(i) by running reward function R(q, o(i))
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
                f"Step {current_step}: Computed rewards/advantages in {reward_duration:.2f}s | Avg reward: {avg_reward:.4f}")
            wandb.log({
                "train_step": current_step,
                "train/reward_computation_duration": reward_duration,
                "train/avg_reward": raw_rewards.mean().item(),
                "train/avg_advantage": advantages.mean().item()
            })

            # 4. tokenizer inputs
            token_start = datetime.now()
            combo = tokenize_prompt_and_output(
                prompt_strs=repeated_prompts,
                output_strs=responses,
                tokenizer=self.tokenizer,
            )
            input_rollouts, labels, response_mask = (
                combo["input_ids"], combo["labels"], combo["response_mask"])
            tokenizer_duration = (datetime.now() - token_start).total_seconds()
            logger.info(
                f"Step {current_step}: Tokenized {len(repeated_prompts)} prompt-response pairs in {tokenizer_duration:.2f}s | Sequence length: {input_rollouts.shape[1]}")
            wandb.log({
                "train_step": current_step,
                "train/tokenization_duration": tokenizer_duration,
                "train/sequence_length": input_rollouts.shape[1]
            })

            # 5. Start train
            epoch_start = datetime.now()
            micro_step_losses = []
            for micro_step in range(self.n_microbatches_per_rollout_batch):
                micro_start = datetime.now()
                current_micro_step = micro_step + 1

                start_idx = micro_step * self.micro_train_batch_size
                end_idx = start_idx + self.micro_train_batch_size

                # prepare micro data
                micro_x = input_rollouts[start_idx:end_idx].to(train_device)
                micro_y = labels[start_idx:end_idx].to(train_device)
                micro_mask = response_mask[start_idx:end_idx].to(train_device)
                micro_rewards = raw_rewards[start_idx:end_idx].to(train_device)
                micro_advantages = advantages[start_idx:end_idx].to(train_device)

                logger.info(
                    f"Step {current_step}, Micro-step {current_micro_step}: Processing batch {start_idx}-{end_idx} (size: {self.micro_train_batch_size})")

                # compute policy log_probs
                policy_log_probs = get_response_log_probs(
                    model=self.policy,
                    input_ids=micro_x,
                    labels=micro_y,
                )["log_probs"]

                # execute micro batch train
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
                    f"Step {current_step}, Micro-step {current_micro_step}: Loss = {loss_value:.4f} | Duration = {micro_duration:.2f}s")
                wandb.log({
                    "grpo_step": current_step,
                    "training/micro_step_loss": loss_value,
                    "training/micro_step_duration": micro_duration,
                    "training/micro_step": current_micro_step
                })

                # do step for each gradient_accumulation_steps
                if (micro_step + 1) % self.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                    self.optimizer.zero_grad()
                    logger.info(
                        f"Step {current_step}: Optimizer step completed (gradient accumulation window finished)")

            avg_epoch_loss = sum(micro_step_losses) / len(micro_step_losses) if micro_step_losses else 0.0
            epoch_duration = (datetime.now() - epoch_start).total_seconds()
            step_duration = (datetime.now() - step_start).total_seconds()
            logger.info(
                f"=== Completed GRPO Step {current_step}/{self.n_grpo_steps} === "
                f"\n Avg loss: {avg_epoch_loss:.4f}"
                f"\n Duration: {epoch_duration:.2f}s"
                f"")
            wandb.log({
                "grpo_step": current_step,
                "training/epoch_loss": avg_epoch_loss,
                "training/epoch_duration": epoch_duration,
                "training/step_duration": step_duration,
            })

            # Set the old policy model πθold ←πθ
            self.old_policy_wrapper.load_policy_into_vllm(self.policy)

            # Evaluate the policy
            if current_step % self.validate_interval == 0:
                self.evaluate(current_step)

        total_train_duration = (datetime.now() - train_start_time).total_seconds()
        logger.info(f"\n=== GRPO Training Complete ==="
                    f"\n  Total steps: {self.n_grpo_steps}"
                    f"\n  Total duration: {total_train_duration:.2f}s ({total_train_duration / 60:.2f} mins)")


if __name__ == "__main__":
    log_init(task_name="grpo")

    # init policy
    model_path = "./data/models/Qwen2.5-Math-1.5B-test"
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