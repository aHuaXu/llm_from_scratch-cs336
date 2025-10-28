import os
import time
from typing import List

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

    flatten: bool = True,
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

        if flatten:
            all_reduce_params_flatten(params)
        else:
            for param in params:
                dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.AVG, async_op=False)

        optimizer.step()

        print(f"step: {step}, rank: {rank}, loss: {loss.item()}, "
              f"params: {[param.sum().item() for param in params]}", flush=True)

    cleanup()

def all_reduce_params_flatten(params: List[torch.Tensor], async_op: bool = False):
    # 1. Collect non-empty gradients
    grads = [param.grad for param in params if param.grad is not None]

    # 2. Flatten using PyTorch's internal utility
    flatten_grad = torch._utils._flatten_dense_tensors(grads)

    # 3. Perform all-reduce on the flattened tensor
    handler = dist.all_reduce(flatten_grad, op=dist.ReduceOp.AVG, async_op=async_op)

    # 4. Unflatten back to original shapes using the matching utility
    unflatten_grads = torch._utils._unflatten_dense_tensors(flatten_grad, grads)

    # 5. Assign unflatten gradients back to parameters
    grad_idx = 0
    for param in params:
        if param.grad is not None:
            param.grad = unflatten_grads[grad_idx]
            grad_idx += 1

    return handler

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
