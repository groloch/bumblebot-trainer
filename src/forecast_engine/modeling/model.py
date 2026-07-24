import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Encoder, EncoderOutput
from .forecast_head import ForecastHead, ForecastOutput
from .policy_head import PolicyHead, PolicyOutput
from .value_head import ValueHead, ValueOutput
from .embedding import Embedding
from ..config.modeling_configs import ModelConfig

from typing import List, Tuple, Optional


class ChessModel(nn.Module):
    def __init__(self, model_config: ModelConfig):
        super().__init__()

        self.embedding = Embedding(model_config.hidden_size)
        self.encoder = Encoder(model_config.encoder_config)

        cls_loss_fn = nn.CrossEntropyLoss(
            ignore_index=-100,
        )
        reg_loss_fn = nn.L1Loss()

        self.policy_head = PolicyHead(model_config.hidden_size, cls_loss_fn)
        self.value_head = ValueHead(model_config.hidden_size, reg_loss_fn)
        self.forecast_head = ForecastHead(model_config.forecast_config, cls_loss_fn)

    def forward(
            self,
            x: torch.Tensor,
            hm: torch.Tensor,
            epsq: torch.Tensor,
            trajectories: Optional[torch.Tensor],
            trajectories_padding_mask: Optional[torch.Tensor],
            target_dict: dict = {}) -> Tuple[EncoderOutput, PolicyOutput, ValueOutput, Optional[ForecastOutput]]:
        x: EncoderOutput = self.embed(x, hm, epsq)

        policy_out: PolicyOutput = self.policy_head(
            x.squares_embeddings,
            target_dict.get('policy', None)
        )
        value_out: ValueOutput = self.value_head(
            x.cls_embedding,
            target_dict.get('value', None)
        )

        if trajectories is not None:
            forecast_out: ForecastOutput = self.forecast_head(
                squares_embeddings=x.squares_embeddings,
                cls_embedding=x.cls_embedding,
                trajectories=trajectories,
                forecast_target=target_dict.get('forecast', None),
                forecast_mask=trajectories_padding_mask,
                loss_mask=target_dict.get('loss_mask', None)
            )
        else:
            forecast_out = None

        return x, policy_out, value_out, forecast_out
    
    def embed(self, x: torch.Tensor, hm: torch.Tensor, epsq: torch.Tensor) -> EncoderOutput:
        return self.encoder(self.embedding(x, hm, epsq))

