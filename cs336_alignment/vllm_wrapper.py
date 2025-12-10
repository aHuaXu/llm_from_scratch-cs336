import os
from typing import List

import torch
from unittest.mock import patch
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from transformers import PreTrainedModel

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
        llm_model = self.inf_vllm.llm_engine.model_executor.driver_worker.model_runner.model
        # 获取vLLM的GPU设备（比如 cuda:0/cuda:1）
        vllm_device = next(llm_model.parameters()).device
        # 获取vLLM的参数类型（99%是 float16，少数是 bf16）
        vllm_dtype = next(llm_model.parameters()).dtype

        # 2. 处理policy权重：跨GPU迁移 + bf16→vLLM指定类型（float16/bf16）
        state_dict = policy.state_dict()
        processed_state_dict = {}
        for k, v in state_dict.items():
            if v is None:
                continue

            # 方案1A: 通过CPU中转进行安全转换
            if v.dtype != vllm_dtype:
                # 先将bfloat16转到CPU的float32，再转到目标类型
                v_cpu_float32 = v.cpu().float()
                processed_v = v_cpu_float32.to(vllm_device).to(vllm_dtype)
            else:
                processed_v = v.to(vllm_device, non_blocking=True)

            processed_state_dict[k] = processed_v

        # 使用load_state_dict而不是load_weights
        llm_model.load_weights(processed_state_dict.items())

        # 3. 清理临时显存（可选，避免OOM）
        del state_dict, processed_state_dict

    def generate(self, prompts: List[str], sampling_params: SamplingParams):
        return self.inf_vllm.generate(prompts, sampling_params, use_tqdm=True)