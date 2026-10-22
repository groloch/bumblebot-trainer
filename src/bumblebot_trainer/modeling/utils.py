from dataclasses import dataclass

import torch

from ..config.modeling_configs import ModelConfig

from typing import Optional


def load_model_state_dict(model, state_dict):
    """Load a state dict into a model, requiring an exact key match."""
    return model.load_state_dict(state_dict, strict=True)


def load_teacher(model, state_dict):
    """Load an EMA teacher checkpoint.
    """
    prefixed = {
        key: value
        for key, value in state_dict.items()
        if key.startswith('module.')
    }
    if 'n_averaged' in state_dict and prefixed and len(prefixed) == len(state_dict) - 1:
        state_dict = {
            key[len('module.'):]: value
            for key, value in prefixed.items()
        }
    return load_model_state_dict(model, state_dict)


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
