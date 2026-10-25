import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Encoder, EncoderOutput
from ...config import CFEncoderConfig
from ...utils import ChessConstants


class FFN(nn.Module):
    """Simple MLP FFN layer. Does not contain a gated activation since
    https://arxiv.org/abs/2409.12272 did not find it to improve performances.
    """
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.linear1 = nn.Linear(hidden_size, intermediate_size)
        self.linear2 = nn.Linear(intermediate_size, hidden_size)
        self.linear2.is_residual_tail = True

        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _x = self.norm(x)

        _x = self.linear1(_x)
        _x = F.silu(_x)
        _x = self.linear2(_x)

        return x + _x


class MHA(nn.Module):
    """Multi-head self attention with smolgen (geometric attention bias)
    that uses flashattn
    """
    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            compressed_dim: int,
            smolgen_dim: int,
            gen_dim: int,
            **kwargs):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj.is_residual_tail = True # init helper
        self.norm = nn.LayerNorm(hidden_size)

        self.smolgen = Smolgen(
            hidden_size=hidden_size,
            compressed_dim=compressed_dim,
            smolgen_dim=smolgen_dim,
            gen_dim=gen_dim,
            num_heads=num_heads
        )

        self.qnorm = nn.LayerNorm(hidden_size)
        self.knorm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, shared_gen: nn.Module, **kwargs) -> torch.Tensor:
        B, N, _ = x.shape

        _x: torch.Tensor = self.norm(x)

        qkv: torch.Tensor = self.qkv(_x)
        q, k, v = qkv.chunk(3, dim=-1)

        q: torch.Tensor = q + kwargs['pos_embeds']
        k: torch.Tensor = k + kwargs['pos_embeds']

        q = self.qnorm(q)
        k = self.knorm(k)

        q = q.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        g = self.smolgen(_x, shared_gen) # B h 64 64

        _x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=g, is_causal=False
        ) # this is a hack to use flash attention with custom attention bias (smolgen)
        _x = _x.transpose(1, 2).reshape(B, N, self.hidden_size)
        _x = self.out_proj(_x)

        return x + _x


class Smolgen(nn.Module):
    """Smolgen: geometric attention bias for chess modeling:
    https://arxiv.org/abs/2409.12272
    """
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
        self.l1 = nn.Linear(compressed_dim, gen_dim * num_heads)
        self.n1 = nn.LayerNorm(gen_dim * num_heads)

    def forward(self, x: torch.Tensor, shared_gen: nn.Module) -> torch.Tensor:
        B = x.size(0)
        x = self.compress(x).mean(dim=1, keepdim=False)
        x = self.n1(F.silu(self.l1(x)))
        x = x.view(B, self.num_heads, self.gen_dim)

        x = shared_gen(x)
        x = x.view(B, self.num_heads, 64, 64)
        return x


class EncoderLayer(nn.Module):
    """Transformer encoder layer with smolgen and layer dropout as per:
    https://arxiv.org/abs/2609.05275
    """
    def __init__(
            self,
            hidden_size: int,
            intermediate_size: int,
            num_heads: int,
            compressed_dim: int,
            smolgen_dim: int,
            gen_dim: int,
            n_layers: int,
            **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads

        self.mha = MHA(
            hidden_size,
            num_heads,
            compressed_dim,
            smolgen_dim,
            gen_dim,
            **kwargs
        )
        self.ffn = FFN(hidden_size, intermediate_size)

        self.max_dropout = 0.4
        self.n_layers = n_layers

    def _select_indices(self, x, layer_idx, progress):
        dropout = (1 - progress) * self.max_dropout * (layer_idx + 1) / self.n_layers
        active = torch.rand(x.shape[0]) >= dropout
        return active.nonzero(as_tuple=True)[0].to(x.device)

    def forward(
            self,
            x: torch.Tensor,
            shared_gen: nn.Module,
            layer_idx: int, **kwargs) -> torch.Tensor:

        if self.training and kwargs.get('layer_dropout', True):
            selected_indices = self._select_indices(x, layer_idx, kwargs['progress'])
            if selected_indices.numel() == 0:
                return x
            selected = torch.index_select(x, 0, selected_indices)
            updated = self.ffn(self.mha(selected, shared_gen, **kwargs))
            return x.index_copy(0, selected_indices, updated)

        return self.ffn(self.mha(x, shared_gen, **kwargs))


class CFCEncoder(Encoder):
    """ChessFormer encoder backbone with smolgen and layer dropout
    """
    def __init__(self, config: CFEncoderConfig, **kwargs):
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
                    n_layers=config.num_layers,
                    **kwargs
                )
                for _ in range(config.num_layers)
            ]
        )

        self.shared_gen = nn.Linear(
            config.gen_dim,
            64*64
        )

        self.shared_posembed = nn.Parameter(
            torch.randn(ChessConstants.CONTEXT_LENGTH, config.hidden_size) * 0.02, requires_grad=True
        )

        self.layer_dropout = True

    def forward(self, x: torch.Tensor, **kwargs) -> EncoderOutput:

        for i, layer in enumerate(self.layers):
            x = layer(
                x,
                self.shared_gen,
                layer_idx=i,
                pos_embeds=self.shared_posembed,
                layer_dropout=self.layer_dropout,
                **kwargs
            )

        return self._pack_output(x)
