from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

from ..config.modeling_configs import EncoderConfig
from ..utils import ChessConstants


@dataclass
class EncoderOutput:
    squares_embeddings: torch.Tensor
    cls_embedding: torch.Tensor
    registers: torch.Tensor


class Encoder(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()

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
        embedding = self.transformer(inputs_embeds=x).last_hidden_state
        squares_embeddings = embedding[:, :64, :]
        cls_embedding = embedding[:, 64, :]
        registers = embedding[:, 65:, :]

        return EncoderOutput(
            squares_embeddings=squares_embeddings,
            cls_embedding=cls_embedding,
            registers=registers
        )
    

class CFEncoder(Encoder):
    def __init__(self):
        pass
