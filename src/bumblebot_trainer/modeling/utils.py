from dataclasses import dataclass

import torch

from typing import Optional


@dataclass
class HeadOutput:
    logits: torch.Tensor
    loss: Optional[torch.Tensor]
