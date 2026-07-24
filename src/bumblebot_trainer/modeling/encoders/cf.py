import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Encoder, EncoderOutput
from ...config import EncoderConfig


class FFN(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.linear1 = nn.Linear(hidden_size, intermediate_size)
        self.linear2 = nn.Linear(intermediate_size, hidden_size)

        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _x = self.linear1(x)
        _x = F.silu(_x)
        _x = self.linear2(_x)
        x = self.norm(x + _x)
        return x


class MHA(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        qkv = self.qkv(x)

        q, k, v = qkv.chunk(3, dim=-1)
        q = q.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        dropout_p = 0.1 if self.training else 0.0
        _x = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=False)

        # Transpose back and flatten heads: (B, L, hidden_size)
        _x = _x.transpose(1, 2).reshape(B, N, self.hidden_size)

        _x = self.out_proj(_x)
        x = self.norm(_x + x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, num_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads

        self.mha = MHA(hidden_size, num_heads)
        self.ffn = FFN(hidden_size, intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mha(x)
        x = self.ffn(x)
        return x


class CFEncoder(Encoder):
    def __init__(self, config: EncoderConfig):
        super().__init__(config)

        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.intermediate_size,
                    num_heads=config.num_heads,
                )
                for _ in range(config.num_layers)
            ]
        )


    def forward(self, x: torch.Tensor) -> EncoderOutput:
        for layer in self.layers:
            x = layer(x)
        return self._pack_output(x)
