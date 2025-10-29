import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def setup(rank: int, world_size: int, backend: str) -> None:
    """
    初始化分布式环境

    Args:
        rank: 当前进程的唯一标识（0到world_size-1）
        world_size: 总进程数
        backend: gloo(cpu) or nccl(gpu)
    """
    # 设置主节点地址和端口（单机多进程用localhost）
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    # 初始化进程组：使用gloo后端（支持CPU通信）
    # 注意：若使用GPU需切换为nccl后端，并确保张量在cuda设备上
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size
    )

def get_device(rank: int, backend: str) -> torch.device:
    """
    根据后端和 rank 选择设备：
    - 若为 nccl 后端：优先使用第 rank 个 GPU，不存在则用第 0 个 GPU
    - 若为 gloo 后端：强制使用 CPU
    """
    if backend == "nccl":
        # 检查 GPU 可用性
        if torch.cuda.is_available():
            # 获取可用 GPU 数量
            num_gpus = torch.cuda.device_count()
            # 选择设备：若 rank 超出可用 GPU 范围，则用第 0 个
            device_id = rank if rank < num_gpus else 0
            return torch.device(f"cuda:{device_id}")
        else:
            # 无 GPU 可用时，强制使用 CPU（尽管 nccl 通常需要 GPU，此处作为兼容处理）
            return torch.device("cpu")
    else:  # gloo 后端
        return torch.device("cpu")



def cleanup() -> None:
    """销毁分布式进程组，释放资源"""
    dist.destroy_process_group()


def distributed_demo(
    rank: int,
    world_size: int,
    tensor_size: int | tuple[int, ...],
    backend: str,
    benchmark_iters: int = 10,
) -> None:
    """
    分布式演示函数：执行all-reduce操作

    Args:
        rank: 当前进程标识
        world_size: 总进程数
        tensor_size: data size
        backend: 通信后端（gloo/nccl）
        benchmark_iters: 正式测试迭代次数
    """
    # 初始化分布式环境
    setup(rank, world_size, backend)

    # 根据后端选择设备（nccl对应GPU，gloo对应CPU）
    device = get_device(rank, backend)
    print(f"rank: {rank}, world_size: {world_size}, device: {device}")
    dtype = torch.float32

    # 生成随机张量（CPU上，若用GPU需加.cuda()）
    data = torch.randn(tensor_size, dtype=dtype, device=device)
    print(f"Rank {rank} | init data: {data}", flush=True)

    # Warmup
    for _ in range(3):
        # 执行all-reduce操作（默认求和，结果广播到所有进程）
        dist.all_reduce(tensor=data, op=dist.ReduceOp.SUM, async_op=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Wait for CUDA kernels to finish
            dist.barrier()            # Wait for all the processes to get here

    # Perform all-reduce
    start_time = time.time()
    for _ in range(benchmark_iters):
        print(f"Rank {rank} | 操作前数据: {data}", flush=True)  # flush=True确保打印顺序
        dist.all_reduce(tensor=data, op=dist.ReduceOp.SUM, async_op=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Wait for CUDA kernels to finish
            dist.barrier()            # Wait for all the processes to get here
        print(f"Rank {rank} | 操作后数据: {data}", flush=True)
    end_time = time.time()

    # 计算平均耗时（秒/次）
    avg_time = (end_time - start_time) / benchmark_iters

    # 3. 聚合所有进程的结果（仅rank=0打印最终统计）
    all_avg_times = [0.0] * world_size
    dist.all_gather_object(all_avg_times, avg_time)  # 收集所有进程的耗时

    if rank == 0:
        # 打印统计结果
        print(f"\n===== 基准测试结果 =====")
        print(f"进程数: {world_size} | 张量形状: {tensor_size} | 数据类型: {dtype} | 后端: {backend}")
        print(f"各进程平均耗时: {[f'{t:.6f}s' for t in all_avg_times]}")
        print(f"全局平均耗时: {sum(all_avg_times) / world_size:.6f}s")
        print(f"最小耗时: {min(all_avg_times):.6f}s | 最大耗时: {max(all_avg_times):.6f}s")

    # 确保进程组被正确销毁
    cleanup()


if __name__ == "__main__":
    # 总进程数（根据硬件资源调整）
    WORLD_SIZE = 4
    TENSOR_SIZE = (1024, )
    BACKEND = "gloo"
    BENCHMARK_ITERS = 10

    # 检查是否支持多进程
    mp.set_start_method("spawn")  # 跨平台安全的启动方式

    # 启动多进程分布式任务
    mp.spawn(
        fn=distributed_demo,
        args=(WORLD_SIZE, TENSOR_SIZE, BACKEND, BENCHMARK_ITERS),
        nprocs=WORLD_SIZE,
        join=True  # 主进程等待所有子进程完成
    )