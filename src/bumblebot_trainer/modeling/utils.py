from dataclasses import dataclass

import torch

from ..config.modeling_configs import ModelConfig

from typing import Optional


@dataclass
class HeadOutput:
    logits: torch.Tensor
    loss: Optional[torch.Tensor]


def ssl_to_pv_model(ssl_model):
    from .ssl import SSLChessModel
    from .model import ChessModel # hack against circular import
    ssl_model: SSLChessModel

    model_config = ModelConfig(
        input_size = ssl_model.config.input_size,
        hidden_size = ssl_model.config.hidden_size,
        intermediate_size = ssl_model.config.intermediate_size,
        encoder_name = ssl_model.config.encoder_name,
        encoder_config = ssl_model.config.encoder_config
    )


    pv_model = ChessModel(model_config)

    pv_model.encoder.load_state_dict(ssl_model.encoder.state_dict().copy())
    pv_model.embedding.load_state_dict(ssl_model.embedding.state_dict().copy())

    return pv_model
