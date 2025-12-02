import torch
from unittest.mock import patch
from vllm import LLM, vllm_set_random_seed
from transformers import PreTrainedModel
import wandb

class VLLMInitializer:
    """
    基于原代码逻辑的 vLLM 初始化类，适配 SFT/RLHF 场景下的模型加载与日志配置
    """

    def __init__(self, model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
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
                dtype=torch.bfloat16,
                enable_prefix_caching=True,
                gpu_memory_utilization=self.gpu_memory_utilization,
            )
        return self.inf_vllm

    def load_policy_into_vllm(self, policy: PreTrainedModel):
        """原 load_policy_into_vinf_vllm 函数逻辑，加载策略模型权重到vLLM"""
        if self.inf_vllm is None:
            raise RuntimeError("请先调用 init_vllm 初始化 vLLM 实例")

        state_dict = policy.state_dict()
        llm_model = self.inf_vllm.llm_engine.model_executor.driver_worker.model_runner.model
        llm_model.load_weights(state_dict.items())

    @staticmethod
    def setup_wandb_metrics():
        """原代码中wandb日志配置逻辑，静态方法直接复用"""
        # Setup wandb metrics
        wandb.define_metric("train_step")  # the x‑axis for training
        wandb.define_metric("eval_step")  # the x‑axis for evaluation
        # everything that starts with train/ is tied to train_step
        wandb.define_metric("train/*", step_metric="train_step")
        # everything that starts with eval/ is tied to eval_step
        wandb.define_metric("eval/*", step_metric="eval_step")