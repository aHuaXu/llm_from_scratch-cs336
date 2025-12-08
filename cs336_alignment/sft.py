from datetime import datetime

import torch
from typing import Literal, Optional, Callable, Dict, List

import wandb
from torch.optim import AdamW
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
from .vllm_wrapper import VLLMWrapper
from .gen_prompt import PromptDataset
from .sft_helper import tokenize_prompt_and_output, get_response_log_probs, sft_microbatch_train_step
from .drgrpo_grader import r1_zero_reward_fn
from .sft_helper import get_model
from .evaluate import evaluate_vllm
from .init import log_init, env_init
from logging import getLogger

env_init()
logger = getLogger(__name__)

class SFT:
    def __init__(
        self,
        policy: PreTrainedModel,  # 核心策略模型（nn.Module/Transformers PreTrainedModel）
        infer_model: VLLMWrapper,  # inference only
        tokenizer: PreTrainedTokenizer,  # 文本tokenizer
        reward_fn: Callable[[str, str], Dict[str, float]] = r1_zero_reward_fn,  # (response, ground_truth) -> reward

        n_sft_steps: int = 200,
        dataset_size: int = 128,    # unique examples for SFT
        learning_rate: float = 1e-5,
        rollout_batch_size: int = 64,
        validate_interval: int = 5,
        gradient_accumulation_steps: int = 4,
    ):
        self.policy = policy
        self.infer_model = infer_model
        self.reward_fn = reward_fn
        self.tokenizer = tokenizer

        self.n_sft_steps = n_sft_steps
        self.rollout_batch_size = rollout_batch_size
        self.learning_rate = learning_rate
        self.validate_interval = validate_interval

        assert rollout_batch_size % gradient_accumulation_steps == 0, \
            "rollout_batch_size must be divisible by number of gradient accumulation steps"
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.micro_batch_size = rollout_batch_size // gradient_accumulation_steps

        self.train_dataset = PromptDataset(
            sample_size=dataset_size,
        )
        self.optimizer = AdamW(
            policy.parameters(),
            lr=learning_rate,
            weight_decay=0.0,
            betas=(0.9, 0.95),
        )

        wandb.config.update({
            "n_sft_steps": n_sft_steps,
            "dataset_size": dataset_size,
            "learning_rate": learning_rate,
            "rollout_batch_size": rollout_batch_size,
            "validate_interval": validate_interval,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "micro_batch_size": self.micro_batch_size,
        })

    def train(self):
        # Initialize training state
        self.policy.train()
        logger.info(
            f"Starting SFT training for {self.n_sft_steps} steps | Batch size: {self.rollout_batch_size} | Gradient accumulation steps: {self.gradient_accumulation_steps}")

        # Track evaluation steps separately
        eval_step_counter = 0
        total_training_start = datetime.now()

        for step in range(self.n_sft_steps):
            step_start = datetime.now()
            current_train_step = step + 1  # 1-based indexing

            # 1. Sample batch data
            prompts, answers, _ = self.train_dataset.train_batch(self.rollout_batch_size)
            logger.debug(
                f"Step {current_train_step}/{self.n_sft_steps}: Sampled {len(prompts)} training examples from dataset")

            # 2. Tokenization step
            tokenized_data = tokenize_prompt_and_output(prompts, answers, self.tokenizer)
            input_ids, labels, response_mask = (
                tokenized_data["input_ids"], tokenized_data["labels"], tokenized_data["response_mask"])
            logger.debug(
                f"Step {current_train_step}/{self.n_sft_steps}: Tokenization complete | Input sequence length: {input_ids.shape[1]}")

            # 3. Gradient accumulation loop
            total_loss = 0.0
            total_entropy = 0.0

            for i in range(self.gradient_accumulation_steps):
                accum_step = i + 1
                start_idx = i * self.micro_batch_size
                end_idx = (i + 1) * self.micro_batch_size

                # Extract micro-batch
                micro_x, micro_y, micro_mask = (
                    input_ids[start_idx:end_idx], labels[start_idx:end_idx], response_mask[start_idx:end_idx])
                logger.debug(
                    f"Step {current_train_step}/{self.n_sft_steps} | Accumulation step {accum_step}/{self.gradient_accumulation_steps}: Processing micro-batch {start_idx}-{end_idx}")

                # Compute log probabilities and entropy
                policy_res = get_response_log_probs(
                    model=self.policy, input_ids=micro_x, labels=micro_y, return_token_entropy=True
                )
                log_probs, entropy = policy_res["log_probs"], policy_res["token_entropy"]

                # Calculate loss for micro-batch
                loss, metadata = sft_microbatch_train_step(
                    log_probs=log_probs,
                    response_mask=micro_mask,
                    gradient_accumulation_steps=self.gradient_accumulation_steps
                )

                # Accumulate metrics
                total_loss += loss.item()
                total_entropy += entropy.mean().item()

                logger.debug(
                    f"Step {current_train_step}/{self.n_sft_steps} | Accumulation step {accum_step}: Micro-batch loss = {loss.item():.4f} | Mean entropy = {entropy.mean().item():.4f}")

            # Calculate average metrics across accumulation steps
            avg_loss = total_loss / self.gradient_accumulation_steps
            avg_entropy = total_entropy / self.gradient_accumulation_steps
            step_duration = (datetime.now() - step_start).total_seconds()

            # Log training metrics
            logger.info(
                f"Step {current_train_step}/{self.n_sft_steps} | "
                f"Average loss: {avg_loss:.4f} | "
                f"Average entropy: {avg_entropy:.4f} | "
                f"Step duration: {step_duration:.2f}s"
            )

            # Log to wandb (removed run check - assuming WandB is always initialized)
            wandb.log({
                "train_step": current_train_step,
                "train/avg_loss": avg_loss,
                "train/avg_entropy": avg_entropy,
                "train/step_duration": step_duration,
                "train/learning_rate": self.optimizer.param_groups[0]['lr']
            })

            # 4. Model evaluation
            if current_train_step % self.validate_interval == 0:
                eval_start = datetime.now()
                eval_step_counter += 1

                logger.info(
                    f"Step {current_train_step}/{self.n_sft_steps}: Starting evaluation (eval step {eval_step_counter})...")

                # Load current policy into inference model
                self.infer_model.load_policy_into_vllm(self.policy)
                logger.debug(
                    f"Step {current_train_step}/{self.n_sft_steps}: Loaded policy model into VLLM inference wrapper")

                # Run evaluation
                format_accuracy, accuracy = evaluate_vllm(self.infer_model)
                eval_duration = (datetime.now() - eval_start).total_seconds()

                # Log evaluation results
                logger.info(
                    f"Evaluation {eval_step_counter} complete | "
                    f"Format accuracy: {format_accuracy:.4f} | "
                    f"Accuracy: {accuracy:.4f} | "
                    f"Evaluation duration: {eval_duration:.2f}s"
                )

                # Log to wandb (removed run check)
                wandb.log({
                    "eval_step": eval_step_counter,
                    "eval/format_accuracy": format_accuracy,
                    "eval/accuracy": accuracy,
                    "eval/duration": eval_duration,
                    "eval/corresponding_train_step": current_train_step
                })

            # 5. Update model parameters
            self.optimizer.step()
            self.optimizer.zero_grad()
            logger.debug(f"Step {current_train_step}/{self.n_sft_steps}: Optimizer step completed | Gradients zeroed")

        # Training completion
        total_training_duration = (datetime.now() - total_training_start).total_seconds()
        logger.info(
            f"SFT training completed! | "
            f"Total steps: {self.n_sft_steps} | "
            f"Total duration: {total_training_duration:.2f}s ({total_training_duration / 60:.2f} mins) | "
            f"Total evaluations performed: {eval_step_counter}"
        )

        # Final wandb log (removed run check)
        wandb.log({
            "train/training_complete": True,
            "train/total_duration_seconds": total_training_duration,
            "train/total_evaluations": eval_step_counter
        })


if __name__ == "__main__":
    log_init(task_name="sft")

    policy, tokenizer, inf_vllm = get_model()
    wandb.watch(policy, log="all")
    logger.info("load model successfully")

    sft = SFT(
        policy=policy,
        infer_model=inf_vllm,
        tokenizer=tokenizer,
    )
    sft.train()

    wandb.finish()