import torch
import torch.nn as nn

from .heads import ValueHead


class PositionClassificationHeads(nn.Module):
    """Classification module to add to an encoder for expert routing.
    """
    ACTIVITY_CLASSES = ('strategical', 'tactical')
    EVALUATION_CLASSES = ('winning', 'drawing', 'losing')

    def __init__(self, hidden_size: int):
        super().__init__()
        self.activity_head = ValueHead(
            64*hidden_size,
            loss_fn=nn.CrossEntropyLoss(),
            output_dim=2,
        )
        self.evaluation_head = ValueHead(
            64*hidden_size,
            loss_fn=nn.CrossEntropyLoss(),
            output_dim=3,
        )

    def forward(self, embeddings: torch.Tensor, activity_target=None, evaluation_target=None):
        B, _, _ = embeddings.shape
        embeddings = embeddings.view(B, -1)
        activity = self.activity_head(embeddings, activity_target)
        evaluation = self.evaluation_head(embeddings, evaluation_target)
        logits = {
            'activity': activity.logits,
            'evaluation': evaluation.logits,
        }
        losses = {
            'activity': activity.loss,
            'evaluation': evaluation.loss,
        }
        return logits, losses
