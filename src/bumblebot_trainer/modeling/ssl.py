from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

from .embedding import Embedding
from .encoders import Encoder, EncoderOutput, build_encoder
from .heads import PolicyHead, PolicyOutput, ValueHead, ValueOutput
from ..config.modeling_configs import PredictorConfig, SSLModelConfig
from ..utils import ChessConstants
from ..data.ssl.utils import SSLConstants


class SSLChessModel(nn.Module):
    """Chess model for JEPA-like training with modulable training objectives.
    This model has two training objectives: predicting the policy (move distribution) and
    the position value.
    Policy is a lc0-style attention policy head.
    Value is a simple MLP regression head over the global (mean) embedding.
    """
    def __init__(self, config: SSLModelConfig):
        super().__init__()
        self.config: SSLModelConfig = config

        self.embedding = Embedding(
            config.input_size,
            config.hidden_size,
            config.intermediate_size
        )
        self.encoder = build_encoder(
            config.encoder_name,
            config=config.encoder_config
        )
        self.policy_head = PolicyHead(
            hidden_size=config.hidden_size,
            loss_fn=nn.CrossEntropyLoss()
        )
        self.value_head = ValueHead(
            hidden_size=config.hidden_size,
            loss_fn=nn.SmoothL1Loss()
        )

    def forward(
            self,
            x: torch.Tensor,
            target: dict[str, torch.Tensor] | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:

        x, x_norm = self.embed(x)

        if target is None:
            return x_norm, None, None

        logits, losses = self.heads_out(x, target)
        return x_norm, logits, losses

    def embed(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Embeds the input tensor and returns embeddings (raw and normalized along the last dimension).

        Args:
            x (torch.Tensor): input tensor of shape (B, 64, hidden_size)

        Returns:
            torch.Tensor: raw square embeddings of shape (B, 64, hidden_size)
            torch.Tensor: normalized square embeddings of shape (B, 64, hidden_size)
        """
        x = self.embedding(x)
        x: EncoderOutput = self.encoder(x)
        sq_emb: torch.Tensor = x.squares_embeddings
        x_norm = nn.functional.normalize(sq_emb, p=2, dim=-1)
        return sq_emb, x_norm

    def heads_out(
            self,
            x: torch.Tensor,
            target: dict[str, torch.Tensor]
            ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Computes the outputs of the heads

        Args:
            x (torch.Tensor): input embeddings of shape (B, 64, hidden_size)
            target (dict[str, torch.Tensor]): policy and value targets

        Returns:
            dict[str, torch.Tensor]: logits for policy and value
            dict[str, torch.Tensor]: losses for policy and value
        """
        policy_out: PolicyOutput = self.policy_head(x, target['policy'])
        value_out: ValueOutput = self.value_head(x.mean(dim=1), target['value'])
        logits = {'policy': policy_out.logits, 'value': value_out.logits}
        losses = {'policy': policy_out.loss, 'value': value_out.loss}
        return logits, losses


class PredictorEmbedding(nn.Module):
    """Custom embedding for moves given to the predictor.
    Each move is made of 6 tokens: initial square, target square, color of the player, piece type,
    piece taken, piece promoted.
    Each of these tokens is embedded separately with 3 levels of hierarchical embeddings: temporal embedding (which
    move is the token part of), role embedding (which token it is within the move), and token embedding (
    which square, piece type or color it is).
    """
    def __init__(self, config: PredictorConfig):
        super().__init__()

        n_squares_tokens = ChessConstants.NUM_SQUARES
        n_piecetype_tokens = ChessConstants.NUM_PIECE_TYPES + 1
        n_color_tokens = ChessConstants.NUM_COLORS

        self.square_embed = nn.Embedding(n_squares_tokens, config.hidden_size)
        self.piecetype_embed = nn.Embedding(n_piecetype_tokens, config.hidden_size)
        self.color_embed = nn.Embedding(n_color_tokens, config.hidden_size)

        self.temporal_embed = nn.Embedding(ChessConstants.MAX_NUMBER_OF_MOVES, config.hidden_size)

        self.role_embed = nn.Embedding(
            SSLConstants.NUM_TOKENS_PER_MOVE,
            config.hidden_size
        )

        self.hidden_size = config.hidden_size
        self.register_buffer('role_ids', torch.arange(SSLConstants.NUM_TOKENS_PER_MOVE))

        self.mask_token_embed = nn.Parameter(torch.randn(config.hidden_size))

    def forward(
            self,
            moves_ids: torch.Tensor,
            dropout: float = 0.0) -> torch.Tensor:
        B, N, _ = moves_ids.shape

        embeds = torch.stack([
            self.square_embed(moves_ids[:, :, 0]),     # from square
            self.square_embed(moves_ids[:, :, 1]),     # to square
            self.color_embed(moves_ids[:, :, 2]),      # turn color
            self.piecetype_embed(moves_ids[:, :, 3]),  # piece moved
            self.piecetype_embed(moves_ids[:, :, 4]),  # piece taken
            self.piecetype_embed(moves_ids[:, :, 5]),  # piece promoted
        ], dim=2)  # (B, N, 6, H)

        if self.training and dropout > 0:
            mask = torch.rand(B, N, 6, device=moves_ids.device) < dropout
            mask_token = self.mask_token_embed.view(1, 1, 1, self.hidden_size).expand(B, N, 6, self.hidden_size)
            embeds = torch.where(mask.unsqueeze(-1), mask_token, embeds)

        temp_ids = torch.arange(N, device=moves_ids.device)
        embeds += self.temporal_embed(temp_ids).view(1, N, 1, self.hidden_size)
        embeds += self.role_embed(self.role_ids).view(1, 1, 6, self.hidden_size)

        return embeds.view(B, N * 6, self.hidden_size)


class Predictor(nn.Module):
    """Predictor for JEPA-like training. This model takes an embedding from the student (64 tokens),
    a sequence of moves (see PredictorEmbedding for the specific format), and tries to predict the
    teacher embeddings of the position after this sequence of move is played.
    The predictor is a simple BERT model with a custom embedding layer for the moves.
    """
    def __init__(self, config: PredictorConfig):
        super().__init__()

        bert_config = BertConfig(
            vocab_size=1,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_layers,
            num_attention_heads=config.num_heads,
            intermediate_size=config.intermediate_size,
            max_position_embeddings=ChessConstants.MAX_NUMBER_OF_MOVES + 64,
            num_labels=1,
            pad_token_id=0,
            bos_token_id=0,
            eos_token_id=0,
        )
        self.embed_embed = nn.Linear(config.hidden_size, config.hidden_size)
        self.moves_embed = PredictorEmbedding(config)

        self.transformer = BertModel(bert_config)

    def forward(
        self,
        x: torch.Tensor,
        moves_ids: torch.Tensor,
        moves_attention_mask: torch.Tensor,
        dropout: float = 0.0,
    ) -> torch.Tensor:
        B, N, D = x.shape

        embedding = self.embed_embed(x)
        moves_embeds = self.moves_embed(moves_ids, dropout)

        inputs_embeds = torch.cat([embedding, moves_embeds], dim=1)

        embed_mask = torch.ones(B, N, dtype=moves_attention_mask.dtype, device=moves_attention_mask.device)
        attention_mask = torch.cat([embed_mask, moves_attention_mask], dim=1)  # (B, N)

        embedding = self.transformer(inputs_embeds=inputs_embeds, attention_mask=attention_mask).last_hidden_state

        prediction = embedding[:, :64, :]
        prediction_norm = nn.functional.normalize(prediction, p=2, dim=-1)
        return prediction, prediction_norm
