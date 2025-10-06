import torch
from torch import nn
from cs336_basics.base_module import (
    LinearLayer,
    EmbeddingLayer,
    RMSNormLayer,
    SwiGLU,
    RotaryPositionalEmbedding,
    MultiHeadAttention,
    softmax,
)

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,

        theta: float | None = None,  # Θ value for the RoPE
        max_seq_len: int | None = None,  # Maximum sequence length that will be inputted
    ):
        """
        :param d_model: Dimensionality of the Transformer block inputs
        :param num_heads: Number of heads to use in multi-head self-attention
        :param d_ff: Dimensionality of the position-wise feed-forward inner layer
        """
        super().__init__()
        self.norm1 = RMSNormLayer(d_model, device=device, dtype=dtype)
        self.norm2 = RMSNormLayer(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadAttention(d_model, num_heads, device, dtype, theta, max_seq_len)
        self.ffn = SwiGLU(d_model, d_ff, device, dtype)

    def load_weights(self, weights: dict[str, torch.Tensor]) -> None:
        state_dict = {
            "attn.Wq.W": weights["attn.q_proj.weight"],
            "attn.Wk.W": weights["attn.k_proj.weight"],
            "attn.Wv.W": weights["attn.v_proj.weight"],
            "attn.Wo.W": weights["attn.output_proj.weight"],
            "norm1.g": weights["ln1.weight"],
            "norm2.g": weights["ln2.weight"],
            "ffn.linear1.W": weights["ffn.w1.weight"],
            "ffn.linear2.W": weights["ffn.w2.weight"],
            "ffn.linear3.W": weights["ffn.w3.weight"],
        }
        self.load_state_dict(state_dict, strict=True)

    # x.shape: (batch_size, seq_len, d_model)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        token_positions = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        # token_positions = token_positions.repeat(x.shape[0], 1)
        y = x + self.attn(self.norm1(x), token_positions)
        return y + self.ffn(self.norm2(y))
    
class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        rope_theta: float | None = None,
    ):
        """
        :param vocab_size:
            The size of the vocabulary, necessary for determining the dimensionality of the token embedding matrix
        :param context_length:
            The maximum context length, necessary for determining the dimensionality of the position embedding matrix
        :param num_layers:
            The number of Transformer blocks to use
        :param d_model:
        :param num_heads:
        :param d_ff:
        :param device:
        :param dtype:
        :param rope_theta:
        """
        super().__init__()
        self.token_embedding = EmbeddingLayer(vocab_size, d_model, device=device, dtype=dtype)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                device=device,
                dtype=dtype,
                theta=rope_theta,
                max_seq_len=context_length,
            ) for _ in range(num_layers)
        ])
        self.norm = RMSNormLayer(d_model, device=device, dtype=dtype)
        self.out_embedding = LinearLayer(d_model, vocab_size, device=device, dtype=dtype)

    def load_weights(self, weights: dict[str, torch.Tensor]) -> None:
        assert weights["token_embeddings.weight"].shape == self.token_embedding.W.shape, \
            f"Token embedding shape mismatch: expected {self.token_embedding.W.shape}, got {weights['token_embeddings.weight'].shape}"
        self.token_embedding.W = nn.Parameter(weights["token_embeddings.weight"])
        for idx, block in enumerate(self.transformer_blocks):
            layer_prefix = f"layers.{idx}."
            layer_weights = {
                k.replace(layer_prefix, ""): v
                for k, v in weights.items()
                if k.startswith(layer_prefix)
            }
            block.load_weights(layer_weights)
        self.norm.g = nn.Parameter(weights["ln_final.weight"])
        self.out_embedding.W = nn.Parameter(weights["lm_head.weight"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.token_embedding(x)
        for block in self.transformer_blocks:
            y = block(y)
        y = self.norm(y)
        return self.out_embedding(y)
