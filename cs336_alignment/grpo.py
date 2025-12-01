import torch
from typing import Literal, Optional, Callable, Dict, List

from torch.optim import AdamW
from transformers import AutoTokenizer, PreTrainedModel
from vllm import LLM, SamplingParams
import copy
from .initializer import VLLMInitializer
from .gen_prompt import PromptDataset

class Grpo:
    def __init__(
        self,
        policy: torch.nn.Module,  # 核心策略模型（nn.Module/Transformers PreTrainedModel）
        old_policy: LLM,    # inference only
        reward_fn: Callable[[str, str], Dict[str, float]],
        raw_questions: List[str],
        tokenizer: AutoTokenizer,  # 文本tokenizer

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
        micro_train_batch_size = train_batch_size // gradient_accumulation_steps

        assert rollout_batch_size % group_size == 0, "rollout_batch_size must be divisible by group_size"
        n_prompts_per_rollout_batch = rollout_batch_size // group_size

        assert train_batch_size >= group_size, "train_batch_size must be greater than or equal to group_size"
        n_microbatches_per_rollout_batch = rollout_batch_size // micro_train_batch_size

        self.sampling_params = SamplingParams(
            temperature=sampling_temperature,
            min_tokens=sampling_min_tokens,
            max_tokens=sampling_max_tokens,
            n=group_size,
            stop=["</answer>"],
            top_p=1.0,
        )
        self.prompt_dataset = PromptDataset(raw_questions)
        self.optimizer = AdamW(
            policy.parameters(),
            lr=learning_rate,
            weight_decay=0.0,
            betas=(0.9, 0.95),
        )

        self.policy = policy
        self.old_policy = old_policy
        self.reward_fn = reward_fn
        self.tokenizer = tokenizer

        self.n_grpo_steps = n_grpo_steps
        self.learning_rate = learning_rate
        self.advantage_eps = advantage_eps
        self.epochs_per_rollout_batch = epochs_per_rollout_batch
        self.loss_type = loss_type
        self.use_std_normalization = use_std_normalization

