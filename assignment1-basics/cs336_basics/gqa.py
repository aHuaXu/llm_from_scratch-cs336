import torch
from torch import nn

from cs336_basics.base_module import LinearLayer, RotaryPositionalEmbedding, scaled_dot_product_attention


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,

        # the following params is not None if you need rope
        theta: float | None = None,  # Θ value for the RoPE
        max_seq_len: int | None = None,  # Maximum sequence length that will be inputted
    ):
        super().__init__()
        assert d_model % num_heads == 0
        assert num_heads % num_kv_heads == 0

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads

        self.rope_layer = None
        if theta is not None:
            assert max_seq_len is not None
            self.rope_layer = RotaryPositionalEmbedding(theta, self.head_dim, max_seq_len, device=device)

        self.W_q = LinearLayer(d_model, d_model, device=device, dtype=dtype)
        self.W_k = LinearLayer(d_model, num_kv_heads * self.head_dim, device=device, dtype=dtype)
        self.W_v = LinearLayer(d_model, num_kv_heads * self.head_dim, device=device, dtype=dtype)
        self.W_o = LinearLayer(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        *batch_dim, seq_len, _ = x.shape

        # Q: (..., seq_len, d_model) -> (..., num_heads, seq_len, head_dim)
        queries = self.W_q(x).reshape(*batch_dim, seq_len, self.num_heads, self.head_dim)
        queries = queries.permute(*range(len(batch_dim)), -2, -3, -1)

        # K, V: (..., seq_len, num_kv_heads * head_dim) -> (..., num_kv_heads, seq_len, head_dim)
        keys = self.W_k(x).reshape(*batch_dim, seq_len, self.num_kv_heads, self.head_dim)
        keys = keys.permute(*range(len(batch_dim)), -2, -3, -1)

        values = self.W_v(x).reshape(*batch_dim, seq_len, self.num_kv_heads, self.head_dim)
        values = values.permute(*range(len(batch_dim)), -2, -3, -1)

        if self.rope_layer is not None:
            queries = self.rope_layer(queries, token_positions)
            keys = self.rope_layer(keys, token_positions)

        # 扩展 KV heads 以匹配 Q heads: (..., num_kv_heads, seq_len, head_dim) -> (..., num_heads, seq_len, head_dim)
        num_groups = self.num_heads // self.num_kv_heads
        keys = keys.repeat_interleave(num_groups, dim=-3)
        values = values.repeat_interleave(num_groups, dim=-3)

        # causal mask
        mask = torch.arange(seq_len, device=x.device)[:, None] >= torch.arange(seq_len, device=x.device)
        for _ in range(len(batch_dim) + 1):
            mask = mask.unsqueeze(0)
        mask = mask.expand(*batch_dim, self.num_heads, seq_len, seq_len)

        # (..., num_heads, seq_len, head_dim)
        out = scaled_dot_product_attention(queries, keys, values, mask)

        # (..., num_heads, seq_len, head_dim) -> (..., seq_len, d_model)
        out = out.permute(*range(len(batch_dim)), -2, -3, -1)
        out = out.reshape(*batch_dim, seq_len, self.d_model)

        return self.W_o(out)
