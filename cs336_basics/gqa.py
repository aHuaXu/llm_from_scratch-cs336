import torch
from torch import nn


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_groups: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,

        # the following params is not None if you need rope
        theta: float | None = None,  # Θ value for the RoPE
        max_seq_len: int | None = None,  # Maximum sequence length that will be inputted
    ):
        super().__init__()
