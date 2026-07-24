import torch
import torch.nn as nn

from .utils import HeadOutput


class ValueOutput(HeadOutput):
    pass


class ValueHead(nn.Module):
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
            mask = target != -1.0
            if mask.any():
                loss = self.loss_fn(nn.functional.sigmoid(value[mask]), target[mask])
            else:
                loss = None
        else:
            loss = None
        return ValueOutput(logits=value, loss=loss)
