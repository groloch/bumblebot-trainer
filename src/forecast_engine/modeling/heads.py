import torch
import torch.nn as nn

from .utils import HeadOutput


class PolicyOutput(HeadOutput):
    pass


class ValueOutput(HeadOutput):
    pass



class PolicyHead(nn.Module):
    """Chessformer policy head from
    https://arxiv.org/pdf/2605.19091
    """
    def __init__(self, hidden_size: int, loss_fn: nn.Module):
        super().__init__()
        self.hidden_size = hidden_size
        self.scale = hidden_size ** -0.5

        self.p_from = nn.Linear(hidden_size, hidden_size)
        self.p_to = nn.Linear(hidden_size, hidden_size)

        self.pk_proj = nn.Linear(hidden_size, 4)

        self.loss_fn = loss_fn

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

        if target is not None:
            loss = self.loss_fn(logits, target)
        else:
            loss = None

        return PolicyOutput(logits=logits, loss=loss)


class ValueHead(nn.Module):
    """Simple MLP value head for chess position evaluation in expected win chance.
    """
    def __init__(self, hidden_size: int, loss_fn: nn.Module):
        super().__init__()
        self.value_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1)
        )

        self.loss_fn = loss_fn

    def forward(self, cls_embedding: torch.Tensor, target: torch.Tensor = None) -> ValueOutput:
        value = self.value_proj(cls_embedding).squeeze(-1)

        if target is not None:
            loss = self.loss_fn(nn.functional.sigmoid(value), target)
        else:
            loss = None
        return ValueOutput(logits=value, loss=loss)
