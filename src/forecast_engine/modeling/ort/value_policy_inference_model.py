import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..encoders import build_encoder
from ..heads import PolicyHead, ValueHead
from ..embedding import Embedding
from ...config.modeling_configs import ModelConfig

from typing import List, Tuple, Optional


class PVInferenceModel(nn.Module):
    """Utility function to convert a model from a training run to a cleaner arch
    used for inference in an engine using ORT.

    This is deprecated and will be removed
    """
    def __init__(self, model_config: ModelConfig):
        super().__init__()
        sys.stderr.write(
            'WARNING: PVInferenceModel is deprecated and will be removed in future versions.\n'
            )

        self.embedding = Embedding(model_config.hidden_size)
        self.encoder = build_encoder(model_config.encoder_name, config=model_config.encoder_config)

        cls_loss_fn = nn.CrossEntropyLoss(
            ignore_index=-100,
        )
        reg_loss_fn = nn.L1Loss()

        self.policy_head = PolicyHead(model_config.hidden_size, cls_loss_fn)
        self.value_head = ValueHead(model_config.hidden_size, reg_loss_fn)

    def forward(
            self,
            x: torch.Tensor,
            hm: torch.Tensor,
            epsq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = x.size(0)

        x = self.embedding(x, hm, epsq)
        x = self.encoder.transformer(x)
        squares_embeddings = x[:, :64, :]
        cls_embedding = x[:, 64, :]

        q = self.policy_head.p_from(squares_embeddings) # B 64 D
        k = self.policy_head.p_to(squares_embeddings) # B 64 D

        attn: torch.Tensor = q @ k.transpose(-2, -1) # B 64 64

        promo_keys = k[:, -8:, :] # B 8 D
        promo_offsets = self.policy_head.pk_proj(promo_keys).permute(0, 2, 1) # B 4 8
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

        attn = attn * self.policy_head.scale

        policy_logits = torch.cat([
            attn.reshape(B, -1),
            promotion_logits.reshape(B, -1)
        ], dim=-1)

        policy = policy_logits.softmax(dim=-1)


        value = self.value_head.value_proj(cls_embedding).squeeze(-1).sigmoid()


        return policy, value
