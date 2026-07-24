from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

from .embedding import Embedding
from .encoders import Encoder, EncoderOutput, build_encoder
from .heads import PolicyHead, PolicyOutput
from ..config.modeling_configs import PredictorConfig, SSLModelConfig
from ..utils import ChessConstants


class SSLChessModel(nn.Module):
    """Chess model for JEPA-like training with modulable square-level training objectives
    """
    def __init__(self, config: SSLModelConfig):
        super().__init__()
        self.embedding = Embedding(config.hidden_size)
        self.encoder = build_encoder(
            config.encoder_name,
            config=config.encoder_config
        )
        self.legalmoves_head = PolicyHead(
            hidden_size=config.hidden_size,
            loss_fn=nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([10]))
        )

    def forward(
            self,
            x: torch.Tensor,
            hm: torch.Tensor,
            epsq: torch.Tensor,
            target: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor | None]:

        x = self.embedding(x, hm, epsq)
        x: EncoderOutput = self.encoder(x)
        sq_emb = x.squares_embeddings
        x = x.cls_embedding
        x = nn.functional.normalize(x, p=2, dim=-1)

        if target is None:
            return x, None, None

        policy_out: PolicyOutput = self.legalmoves_head(sq_emb, target)
        logits = policy_out.logits
        loss = policy_out.loss
        return x, logits, loss


class Predictor(nn.Module):
    def __init__(self, config: PredictorConfig):
        super().__init__()

        bert_config = BertConfig(
            vocab_size=ChessConstants.NUM_POLICY_CLASSES + 1,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_layers,
            num_attention_heads=config.num_heads,
            intermediate_size=config.intermediate_size,
            max_position_embeddings=ChessConstants.MAX_NUMBER_OF_MOVES + 1,
            num_labels=None,
            pad_token_id=ChessConstants.NUM_POLICY_CLASSES,
            bos_token_id=0,
            eos_token_id=0,
        )
        self.embed_embed = nn.Linear(config.hidden_size, config.hidden_size)

        self.transformer = BertModel(bert_config)

    def forward(
        self,
        x: torch.Tensor,
        moves_ids: torch.Tensor,
        moves_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, D = x.shape
        B, Nm = moves_ids.shape
        N = Nm + 1

        embedding = self.embed_embed(x).unsqueeze(1)
        moves_embeds = self.transformer.embeddings.word_embeddings(moves_ids)

        inputs_embeds = torch.cat([embedding, moves_embeds], dim=1)

        if moves_attention_mask is None:
            moves_attention_mask = (moves_ids != self.transformer.config.pad_token_id).long()

        embed_mask = torch.ones(B, 1, dtype=moves_attention_mask.dtype, device=moves_attention_mask.device)
        attention_mask = torch.cat([embed_mask, moves_attention_mask], dim=1)  # (B, N)

        embedding = self.transformer(inputs_embeds=inputs_embeds, attention_mask=attention_mask).last_hidden_state

        prediction = embedding[:, 0, :]
        prediction = nn.functional.normalize(prediction, p=2, dim=-1)
        return prediction