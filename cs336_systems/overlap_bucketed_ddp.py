import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from .naive_ddp import (
    all_reduce_params_flatten
)

def get_param_size(param: torch.Tensor) -> float:
    byte_size = param.numel()*param.element_size()
    return byte_size/1024/1024  # MB

class DDPBucketed:
    def __init__(self, module: torch.nn.Module, bucket_size_mb: float):
        """
            Given an instantiated
            PyTorch nn.Module to be parallelized, construct a DDP container that will handle gradient synchronization across ranks.
            Gradient synchronization should be bucketed, with each bucket holding at most bucket_size_mb of parameters.
        """
        self.module = module
        self.async_handlers = []
        self.bucket_size_mb = bucket_size_mb

        self.current_bucket = []
        self.current_bucket_size = 0

        # hook_handlers = []
        for name, param in self.module.named_parameters():
            param.name = name
            param.register_post_accumulate_grad_hook(self._all_reduce_grad_hook(param))


    def forward(self, *inputs, **kwargs):
        """
            Calls the wrapped module’s forward() method with the provided positional and keyword arguments.
        """
        return self.module.forward(*inputs, **kwargs)


    def finish_gradient_synchronization(self):
        """
            When called, wait for asynchronous communication calls to be queued on GPU.
        """
        if self.current_bucket:
            self._sync_grad_bucket()

        for handler in self.async_handlers:
            handler.wait()
        self.async_handlers.clear()

    def _all_reduce_grad_hook(self, param: torch.Tensor):
        grad_size = get_param_size(param.grad)
        if self.current_bucket_size + grad_size > self.bucket_size_mb:
            self._sync_grad_bucket()

        self.current_bucket_size += grad_size
        self.current_bucket.append(param)

    def _sync_grad_bucket(self):
        handler = all_reduce_params_flatten(self.current_bucket, async_op=True)
        self.async_handlers.append(handler)