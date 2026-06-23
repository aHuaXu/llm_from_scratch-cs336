import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from .all_reduce_demo import (
    setup, get_device, cleanup
)


class DDPIndividualParameters:
    def __init__(self, module: torch.nn.Module):
        """
            Given an instantiated PyTorch nn.Module to be parallelized,
            construct a DDP container that will handle gradient synchronization across ranks.
        """
        self.module = module
        self.async_handlers = []

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
        for handler in self.async_handlers:
            handler.waite()
        self.async_handlers.clear()

    def _all_reduce_grad_hook(self, param: torch.Tensor):
        handler = dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, async_op=True)
        self.async_handlers.append(handler)
