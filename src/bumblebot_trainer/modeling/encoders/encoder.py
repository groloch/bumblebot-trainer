from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

from ...config.modeling_configs import EncoderConfig
from ...utils import ChessConstants


@dataclass
class EncoderOutput:
    squares_embeddings: torch.Tensor
    cls_embedding: torch.Tensor


class Encoder(nn.Module):
    """Abstract class for encoders. This helps experiment with several
    architectures (Bert, CNNs, Chessformers, ...)
    """
    def __init__(self, config: EncoderConfig):
        super().__init__()

        self.config = config

    def _pack_output(self, latents) -> EncoderOutput:
        """Packs the output into square-level tokens latents, and a global CLS-like
        latent (mean of tokens latents).
        """
        cls_embed = latents.mean(dim=1, keepdim=False)
        return EncoderOutput(
            squares_embeddings=latents,
            cls_embedding=cls_embed,
        )


class BertEncoder(Encoder):
    """Simplest encoder implementation as a Bert encoder from the transformers library.
    """
    def __init__(self, config: EncoderConfig):
        super().__init__(config)

        bert_config = BertConfig(
            vocab_size=1,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_layers,
            num_attention_heads=config.num_heads,
            intermediate_size=config.intermediate_size,
            max_position_embeddings=ChessConstants.CONTEXT_LENGTH,
            num_labels=None,
            pad_token_id=0,
            bos_token_id=0,
            eos_token_id=0,
        )

        self.transformer = BertModel(bert_config)

    def forward(self, x: torch.Tensor) -> EncoderOutput:
        latents = self.transformer(inputs_embeds=x).last_hidden_state

        return self._pack_output(latents)


class CNNEncoder(Encoder):
    pass


class CFEncoder(Encoder):
    pass
