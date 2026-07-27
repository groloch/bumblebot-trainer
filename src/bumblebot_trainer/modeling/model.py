import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import Encoder, EncoderOutput, build_encoder
from .heads import PolicyHead, PolicyOutput, ValueHead, ValueOutput
from .embedding import Embedding
from ..config.modeling_configs import ModelConfig


class ChessModel(nn.Module):
    """Modulable chess model with embedding, encoder, and policy/value heads.
    """
    def __init__(self, model_config: ModelConfig):
        super().__init__()

        self.embedding = Embedding(
            model_config.input_size,
            model_config.hidden_size,
            model_config.intermediate_size
        )
        self.encoder: Encoder = build_encoder(
            model_config.encoder_name,
            config=model_config.encoder_config
        )

        cls_loss_fn = nn.CrossEntropyLoss(
            ignore_index=-100,
        )
        reg_loss_fn = nn.functional.smooth_l1_loss

        self.policy_head = PolicyHead(model_config.hidden_size, cls_loss_fn)
        self.value_head = ValueHead(model_config.hidden_size, reg_loss_fn)

    def forward(
            self,
            x: torch.Tensor,
            target_dict: dict = {}) -> tuple[EncoderOutput, PolicyOutput, ValueOutput]:
        x: EncoderOutput = self.embed(x)

        policy_out: PolicyOutput = self.policy_head(
            x.squares_embeddings,
            target_dict.get('policy', None)
        )
        value_out: ValueOutput = self.value_head(
            x.cls_embedding,
            target_dict.get('value', None)
        )

        return x, policy_out, value_out

    def embed(self, x: torch.Tensor) -> EncoderOutput:
        return self.encoder(self.embedding(x))

