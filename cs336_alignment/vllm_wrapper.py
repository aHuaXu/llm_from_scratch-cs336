import os
from typing import List

import torch
from unittest.mock import patch
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from transformers import PreTrainedModel
import wandb

class VLLMWrapper:
    """
    基于原代码逻辑的 vLLM 初始化类，适配 SFT/RLHF 场景下的模型加载与日志配置
    """

    def __init__(self, model_id: str, device: str, seed: int = 666, gpu_memory_utilization: float = 0.85):
        self.model_id = model_id
        self.device = device
        self.seed = seed
        self.gpu_memory_utilization = gpu_memory_utilization
        self._init_vllm()

    def _init_vllm(self) -> LLM:
        """原 init_vllm 函数逻辑，初始化vLLM引擎"""
        vllm_set_random_seed(self.seed)

        # 原代码中的两个补丁
        world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
        profiling_patch = patch(
            "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
            return_value=None
        )

        with world_size_patch, profiling_patch:
            self.inf_vllm = LLM(
                model=self.model_id,
                device=self.device,
                dtype=torch.float16,
                enable_prefix_caching=False,
                gpu_memory_utilization=self.gpu_memory_utilization,
                tensor_parallel_size=1,
                disable_custom_all_reduce=True,
            )
        return self.inf_vllm

    def load_policy_into_vllm(self, policy: PreTrainedModel):
        """原 load_policy_into_vinf_vllm 函数逻辑，加载策略模型权重到vLLM"""
        if self.inf_vllm is None:
            raise RuntimeError("请先调用 init_vllm 初始化 vLLM 实例")

        state_dict = policy.state_dict()
        llm_model = self.inf_vllm.llm_engine.model_executor.driver_worker.model_runner.model
        llm_model.load_weights(state_dict.items())

    def generate(self, prompts: List[str], sampling_params: SamplingParams):
        return self.inf_vllm.generate(prompts, sampling_params, use_tqdm=True)