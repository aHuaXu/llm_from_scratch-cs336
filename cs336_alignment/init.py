import logging
import os

import wandb
from datetime import datetime
from logging import Logger, getLogger

def log_init(
    task_name: str,
):
    wandb.init(
        project="cs336_assignment5",
        name=task_name + datetime.now().strftime("%Y%m%d_%H%M%S"),
    )

    # Setup wandb metrics
    wandb.define_metric("train_step")  # the x‑axis for training
    wandb.define_metric("eval_step")  # the x‑axis for evaluation
    # everything that starts with train/ is tied to train_step
    wandb.define_metric("train/*", step_metric="train_step")
    # everything that starts with eval/ is tied to eval_step
    wandb.define_metric("eval/*", step_metric="eval_step")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler()  # 输出到控制台
        ]
    )

def env_init():
    os.environ["CUDA_VISIBLE_DEVICES"] = "6, 7"  # 子进程内屏蔽其他 GPU
    # os.environ["VLLM_ATTENTION_BACKEND"] = "torch"

train_device = "cuda:1"
eval_device = "cuda:0"