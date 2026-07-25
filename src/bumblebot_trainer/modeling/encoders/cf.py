import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Encoder, EncoderOutput
from ...config import CFEncoderConfig


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
    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            compressed_dim: int,
            smolgen_dim: int,
            gen_dim: int):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

        self.smolgen = Smolgen(
            hidden_size=hidden_size,
            compressed_dim=compressed_dim,
            smolgen_dim=smolgen_dim,
            gen_dim=gen_dim,
            num_heads=num_heads
        )

    def forward(self, x: torch.Tensor, shared_gen: nn.Module) -> torch.Tensor:
        B, N, _ = x.shape
        qkv = self.qkv(x)

        q, k, v = qkv.chunk(3, dim=-1)
        q = q.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        g = self.smolgen(x, shared_gen) # B h 64 64

        dropout_p = 0.1 if self.training else 0.0
        _x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=g, dropout_p=dropout_p, is_causal=False
        ) # this is a hack to use flash attention with custom attention bias (smolgen)

        _x = _x.transpose(1, 2).reshape(B, N, self.hidden_size)

        _x = self.out_proj(_x)
        x = self.norm(_x + x)
        return x


class Smolgen(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            compressed_dim: int,
            smolgen_dim: int,
            gen_dim: int,
            num_heads: int):
        super().__init__()

        self.hidden_size = hidden_size
        self.compressed_dim = compressed_dim
        self.smolgen_dim = smolgen_dim
        self.num_heads = num_heads
        self.gen_dim = gen_dim

        self.compress = nn.Linear(hidden_size, compressed_dim)
        self.l1 = nn.Linear(compressed_dim * 64, smolgen_dim)
        self.n1 = nn.LayerNorm(smolgen_dim)
        self.l2 = nn.Linear(smolgen_dim, gen_dim * num_heads)
        self.n2 = nn.LayerNorm(gen_dim * num_heads)

    def forward(self, x: torch.Tensor, shared_gen: nn.Module) -> torch.Tensor:
        B = x.size(0)
        x = self.compress(x).view(B, -1)
        x = self.n1(F.silu(self.l1(x)))

        x = self.n2(F.silu(self.l2(x)))
        x = x.view(B, self.num_heads, self.gen_dim)

        x = shared_gen(x)
        x = x.view(B, self.num_heads, 64, 64)
        return x


class EncoderLayer(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            intermediate_size: int,
            num_heads: int,
            compressed_dim: int,
            smolgen_dim: int,
            gen_dim: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads

        self.mha = MHA(
            hidden_size,
            num_heads,
            compressed_dim,
            smolgen_dim,
            gen_dim
        )
        self.ffn = FFN(hidden_size, intermediate_size)

    def forward(self, x: torch.Tensor, shared_gen: nn.Module) -> torch.Tensor:
        x = self.mha(x, shared_gen)
        x = self.ffn(x)
        return x


class CFEncoder(Encoder):
    def __init__(self, config: CFEncoderConfig):
        super().__init__(config)

        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.intermediate_size,
                    num_heads=config.num_heads,
                    compressed_dim=config.compressed_dim,
                    smolgen_dim=config.smolgen_dim,
                    gen_dim=config.gen_dim,
                )
                for _ in range(config.num_layers)
            ]
        )

        self.shared_gen = nn.Linear(
            config.gen_dim,
            64*64
        )

    def forward(self, x: torch.Tensor) -> EncoderOutput:
        for layer in self.layers:
            x = layer(x, self.shared_gen)
        return self._pack_output(x)
