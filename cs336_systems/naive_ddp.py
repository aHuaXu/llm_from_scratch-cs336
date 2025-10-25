import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from .all_reduce_demo import (
    setup, get_device, cleanup
)
import torch.nn.functional as F

def naive_data_parallelism(
    rank: int,
    world_size: int,
    data: torch.Tensor,
    num_layers: int,
    num_steps: int,
    backend: str,
):
    setup(rank, world_size, backend)
    torch.manual_seed(66)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(66)

    device = get_device(rank, backend)

    batch_size, num_size = data.shape
    assert batch_size % world_size == 0
    local_batch_size = batch_size // world_size
    start_index, end_index = rank * local_batch_size, (rank + 1) * local_batch_size
    local_data = data[start_index:end_index]

    params = [torch.randn(num_size, num_size, requires_grad=True).to(device) for _ in range(num_layers)]
    optimizer = torch.optim.AdamW(params, lr=1e-3)

    for step in range(num_steps):
        optimizer.zero_grad()

        x = local_data
        for param in params:
            x = x * param
            x = F.relu(x)
        loss = x.square().mean()

        loss.backward()

        for param in params:
            dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.AVG, async_op=False)

        optimizer.step()

        print(f"step: {step}, rank: {rank}, loss: {loss.item()}, "
              f"params: {[param.sum().item() for param in params]}", flush=True)

    cleanup()

if __name__ == "__main__":
    # 总进程数（根据硬件资源调整）
    WORLD_SIZE = 4
    NUM_LAYERS = 2
    NUM_STEPS = 10
    BACKEND = "gloo"
    DATA = torch.randn(128, 512)

    # 检查是否支持多进程
    mp.set_start_method("spawn")  # 跨平台安全的启动方式

    # 启动多进程分布式任务
    mp.spawn(
        fn=naive_data_parallelism,
        args=(WORLD_SIZE, DATA, NUM_LAYERS, NUM_STEPS, BACKEND),
        nprocs=WORLD_SIZE,
        join=True  # 主进程等待所有子进程完成
    )
