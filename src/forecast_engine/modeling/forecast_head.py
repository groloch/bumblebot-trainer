from dataclasses import dataclass

import torch
import torch.nn as nn

from .utils import HeadOutput
from ..utils import ChessConstants, ForecastVocabulary
from ..config.modeling_configs import ForecastConfig

from typing import List, Optional


def _create_attn_mask(forecast_depth):
        # Mask to prevent attention between independent trajectories
        # Example: predict next locations for a knight on g1
        # The g1 (idx = 6) token for all steps will only attend to the whole inital board, and
        # the g1 tokens of each step, but not to the other tokens of the other steps

        # Each trajectory should be a forecast (where do I want the knight on g1 to go ideally)
        # and should not be influenced by the other trajectories

        # It should also not be causal: of we want the knight on g1 to go to f4
        # we could predict f4 first (at step2) and then decide between e2 or h3 at step1

        # On another note, board tokens should be able to attend to each trajectory, to refine the board
        # embeddings iteratively

        total_tokens = 65 + 64 * forecast_depth
        mask = torch.zeros((total_tokens, total_tokens), dtype=torch.bool)
        
        # mask[:65, :65] = True
        # mask[:65, :] = True

        for sq in range(64):
            traj_idx = [65 + step * 64 + sq for step in range(forecast_depth)]

            for idx in traj_idx:
                mask[idx, :65] = True
                mask[idx, traj_idx] = True

        mask = torch.where(mask, 0.0, -torch.inf)

        return mask


@dataclass
class ForecastOutput(HeadOutput):
    horizon_logits: torch.Tensor
    horizon_loss: torch.Tensor


class ForecastHead(nn.Module):
    def __init__(self, config: ForecastConfig, loss_fn: nn.Module):
        super().__init__()
        self.forecast_depth = config.forecast_depth

        self.loss_fn = loss_fn

        compressed_dim = config.hidden_size // 4
        self.compressed_dim = compressed_dim

        self.compress_embedding = nn.Linear(config.hidden_size, compressed_dim)

        attn_mask = _create_attn_mask(config.forecast_depth)
        self.register_buffer('attn_mask', attn_mask)
        self.attn_mask: torch.Tensor

        self.nhead = 8
        self.denoiser = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=compressed_dim,
                nhead=self.nhead,
                dim_feedforward=compressed_dim * 2,
                activation='gelu',
                batch_first=True
            ),
            num_layers=4
        )
        self.f_embedding = nn.Embedding(
            ForecastVocabulary.HORIZON_OFFSET,
            compressed_dim
        )
        self.h_embedding = nn.Embedding(3, compressed_dim)
        self.mask_embedding = nn.Parameter(torch.randn(compressed_dim) * 0.02)

        self.n_classes = ForecastVocabulary.HORIZON_OFFSET

        self.positional_encoding = nn.Parameter(
            torch.randn(ForecastVocabulary.PER_HORIZON_CTX_LENGTH, compressed_dim) * 0.02
        )
        self.temporal_encoding = nn.Parameter(
            torch.randn(self.forecast_depth, compressed_dim) * 0.02
        )

    def forward(
            self,
            squares_embeddings: torch.Tensor,
            cls_embedding: torch.Tensor,
            trajectories: torch.Tensor,
            forecast_target: torch.Tensor,
            forecast_mask: torch.Tensor,
            loss_mask: torch.Tensor = None) -> List[ForecastOutput]:
        # Board latent from the encoder:
        # squares_embeddings: B, 64, D
        # cls_embeddings: B, D

        # To denoise: (C = 69, 64 squares + 1 taken logit + 4 promotion logits)
        # trajectories: B, 64*forecast_depth, C

        # Targets
        # forecast_target: B, forecast_depth, 64, 69 (squares to go to)
        # horizon_target: B, forecast_depth, 64, 3 (when to go there)
        # forecast_mask: B, forecast_depth, 64

        tokens = torch.cat([squares_embeddings, cls_embedding.unsqueeze(1)], dim=1)
        compressed = self.compress_embedding(tokens) # B 65 C

        B = squares_embeddings.size(0)

        attn_mask = self.attn_mask.unsqueeze(0).expand(B, -1, -1).clone()
        attn_mask[:, :, 65:] = attn_mask[:, :, 65:].masked_fill(~forecast_mask.unsqueeze(1), -torch.inf)
        attn_mask = attn_mask.repeat_interleave(self.nhead, dim=0)

        mask_token_id = ForecastVocabulary.MASK_TOKEN_ID(self.forecast_depth)
        is_mask = (trajectories == mask_token_id)
        safe_trajectories = torch.where(is_mask, torch.zeros_like(trajectories), trajectories)
        
        trajectories_ = self.f_embedding(safe_trajectories % ForecastVocabulary.HORIZON_OFFSET) # B, 64*forecast_depth, C
        trajectories_emb = trajectories_ + self.h_embedding(safe_trajectories // ForecastVocabulary.HORIZON_OFFSET)
        trajectories_emb = torch.where(is_mask.unsqueeze(-1), self.mask_embedding, trajectories_emb)

        combined_pos = (
            self.temporal_encoding.unsqueeze(1) +
            self.positional_encoding.unsqueeze(0)
        ).view(-1, self.temporal_encoding.size(-1))
        trajectories = trajectories_emb + combined_pos.unsqueeze(0)

        input_embeds = torch.cat([
            compressed,
            trajectories
        ], dim=1) # B (65 + 64*forecast_depth) C

        denoised = self.denoiser(
            input_embeds,
            mask=attn_mask,
            is_causal=False
        ) # B (65 + 64*forecast_depth) C

        forecast_logits: torch.Tensor
        forecast_logits = denoised[:, 65:, :] @ self.f_embedding.weight.t() # B (64*forecast_depth) C

        horizon_logits: torch.Tensor
        horizon_logits = denoised[:, 65:, :] @ self.h_embedding.weight.t() # B (64*forecast_depth) 3

        if forecast_target is not None:
            flat_mask = loss_mask.view(-1)
            forecast_loss = self.loss_fn(
                forecast_logits.reshape(-1, self.n_classes)[flat_mask],
                forecast_target.reshape(-1)[flat_mask] % ForecastVocabulary.HORIZON_OFFSET
            )
            horizon_loss = self.loss_fn(
                horizon_logits.reshape(-1, 3)[flat_mask],
                forecast_target.reshape(-1)[flat_mask] // ForecastVocabulary.HORIZON_OFFSET
            )
        else:
            forecast_loss = None
            horizon_loss = None
        return ForecastOutput(
            logits=forecast_logits,
            loss=forecast_loss,
            horizon_logits=horizon_logits,
            horizon_loss=horizon_loss
        )
