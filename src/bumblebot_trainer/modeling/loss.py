import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueLoss(nn.Module):
    """Combined value loss over a single shared softmax.

    Given the raw value-head logits (B, 3) interpreted as (win, draw, loss):
    - 'value': cross-entropy against the WDL target (soft 3-vector, or class
      indices for classification-style heads),
    - 'qvalue': MAE between the q target and the predicted w - l, i.e.
      probs[..., 0] - probs[..., -1].

    forward(logits, value_target=None, qvalue_target=None) -> dict[str, Tensor]
    Both targets must be row-aligned with `logits`. Missing/empty targets are
    simply omitted from the returned dict. Computation is done in float32 for
    stability under bf16 autocast.
    """
    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(
            self,
            logits: torch.Tensor,
            value_target: torch.Tensor = None,
            qvalue_target: torch.Tensor = None,
            ) -> dict[str, torch.Tensor]:
        has_value = value_target is not None and value_target.shape[0] > 0
        has_qvalue = qvalue_target is not None and qvalue_target.shape[0] > 0
        if not (has_value or has_qvalue):
            return {}

        probs = F.softmax(logits.float(), dim=-1)

        losses = {}
        if has_value:
            # keep class-index (long) targets as-is; cast soft targets to float32
            value_target = value_target if not torch.is_floating_point(value_target) \
                else value_target.float()
            losses['value'] = self.ce(logits.float(), value_target)
        if has_qvalue:
            q_pred = probs[..., 0] - probs[..., -1]
            losses['qvalue'] = F.l1_loss(q_pred, qvalue_target.float().reshape(-1))
        return losses
