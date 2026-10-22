import math

import torch
import torch.nn as nn

STD = 0.02
VALUE_FINAL_STD = 1e-3


def init_weights(model: nn.Module, *, n_layers: int, std: float = STD) -> None:
    tail_scale = math.sqrt(1.0 / (2.0 * n_layers))

    for module in model.modules():
        if isinstance(module, nn.Linear):
            if getattr(module, 'is_value_final', False):
                nn.init.normal_(module.weight, mean=0.0, std=VALUE_FINAL_STD)
            elif getattr(module, 'is_residual_tail', False):
                nn.init.normal_(module.weight, mean=0.0, std=std * tail_scale)
            else:
                nn.init.normal_(module.weight, mean=0.0, std=std)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    for name, parameter in model.named_parameters():
        if name.endswith('mask_token_embed'):
            nn.init.normal_(parameter, mean=0.0, std=std)
