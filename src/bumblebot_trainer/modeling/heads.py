from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import HeadOutput


class PolicyOutput(HeadOutput):
    pass


@dataclass
class ValueOutput(HeadOutput):
    qvalue_loss: torch.Tensor = None



class PolicyHead(nn.Module):
    """Chessformer policy head from
    https://arxiv.org/pdf/2605.19091
    """
    def __init__(self, hidden_size: int, loss_fn: nn.Modul):
        super().__init__()
        self.hidden_size = hidden_size
        self.scale = hidden_size ** -0.5

        self.p_from = nn.Linear(hidden_size, hidden_size)
        self.p_to = nn.Linear(hidden_size, hidden_size)

        self.pk_proj = nn.Linear(hidden_size, 4)

        self.loss_fn = loss_fn

        self.t_proj = nn.Linear(hidden_size, 1)

    def forward(
            self,
            squares_embeddings: torch.Tensor,
            target: torch.Tensor = None) -> PolicyOutput:
        B = squares_embeddings.shape[0]

        q: torch.Tensor
        k: torch.Tensor

        q = self.p_from(squares_embeddings) # B 64 D
        k = self.p_to(squares_embeddings) # B 64 D

        attn: torch.Tensor = q @ k.transpose(-2, -1) * self.scale # B 64 64

        promo_keys = k[:, -8:, :] # B 8 D
        promo_offsets = self.pk_proj(promo_keys).permute(0, 2, 1) # B 4 8
        promo_offsets = promo_offsets[:, :3, :] + promo_offsets[:, 3:4, :]
        q_promo_logits = attn[:, -16:-8, -8:] # B 8 8
        r_promo_logits = q_promo_logits + promo_offsets[:, 0:1, :] # B 8 8
        b_promo_logits = q_promo_logits + promo_offsets[:, 1:2, :] # B 8 8
        n_promo_logits = q_promo_logits + promo_offsets[:, 2:3, :] # B 8 8

        promotion_logits = torch.stack([
            r_promo_logits,
            b_promo_logits,
            n_promo_logits
        ], dim=-1) # B 8 8 3

        logits = torch.cat([
            attn.reshape(B, -1),
            promotion_logits.reshape(B, -1)
        ], dim=-1)

        logits = logits / self.t_proj(squares_embeddings.mean(dim=1, keepdim=False))

        if target is not None:
            loss = self.loss_fn(logits, target)
        else:
            loss = None

        return PolicyOutput(logits=logits, loss=loss)


class ValueHead(nn.Module):
    """Attention-pooled value head for chess position evaluation in expected win chance.
    The encoder square embeddings are used directly as keys and values (no projection);
    a single query (projection of the mean embedding, whose bias acts as a learned
    query) scores the squares, and the attention-weighted sum is read out by an MLP.
    """
    def __init__(self, hidden_size: int, loss_fn: nn.Module, output_dim: int = 3):
        super().__init__()
        self.hidden_size = hidden_size
        self.scale = hidden_size ** -0.5

        self.vq_proj = nn.Linear(hidden_size, hidden_size)

        self.value_proj = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.SiLU(),
            nn.Linear(128, output_dim)
        )
        self.value_proj[2].is_value_final = True # init helper

        self.loss_fn = loss_fn

    def forward(
            self,
            squares_embeddings: torch.Tensor,
            target: torch.Tensor = None,
            qvalue_target: torch.Tensor = None,
            ) -> ValueOutput:
        # squares_embeddings: (B, 64, D); raw encoder embeddings are k and v directly
        q = self.vq_proj(squares_embeddings.mean(dim=1)) # B D; the bias is the learned query
        q = q.view(-1, 1, 1, self.hidden_size) # B 1 1 D: single query token
        kv = squares_embeddings.unsqueeze(1) # B 1 64 D
        pooled = F.scaled_dot_product_attention(q, kv, kv, scale=self.scale) # B 1 1 D
        pooled = pooled.reshape(-1, self.hidden_size) # B D

        value = self.value_proj(pooled)

        if target is not None or qvalue_target is not None:
            losses = self.loss_fn(value, target, qvalue_target)
            loss = losses.get('value')
            qvalue_loss = losses.get('qvalue')
        else:
            loss = None
            qvalue_loss = None
        return ValueOutput(logits=value, loss=loss, qvalue_loss=qvalue_loss)


class SquareHead(nn.Module):
    """Simple MLP head for square-level tasks
    """
    def __init__(self, hidden_size: int, output_dim: int, loss_fn: nn.Module):
        super().__init__()

        self.o_proj = nn.Linear(hidden_size, output_dim)
        self.output_dim = output_dim
        self.loss_fn = loss_fn

    def forward(self, squares_embeddings: torch.Tensor, target: torch.Tensor = None) -> HeadOutput:
        logits: torch.Tensor = self.o_proj(squares_embeddings)

        if target is not None:
            loss = self.loss_fn(
                logits.reshape(-1, self.output_dim),
                target.reshape(-1)
            )
        else:
            loss = None

        return HeadOutput(logits=logits, loss=loss)
