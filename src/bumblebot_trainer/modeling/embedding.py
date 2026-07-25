import torch
import torch.nn as nn

from ..utils import ChessConstants


class Embedding(nn.Module):
    """Simple implementation of an embedding layer with learnt positional encodings and a ffn layer
    after embedding.
    """
    def __init__(self, input_size: int, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.embedding = nn.Linear(
            in_features=input_size, out_features=hidden_size
        )
        self.embed_norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size),
            nn.LayerNorm(hidden_size)
        )
        self.positional_encoding = nn.Parameter(
            torch.zeros(ChessConstants.CONTEXT_LENGTH, hidden_size), requires_grad=True
        )
        self.out_norm = nn.LayerNorm(hidden_size)

    def forward(self, tokens):
        x = self.embedding(tokens) + self.positional_encoding
        x = nn.functional.silu(x)
        x = self.embed_norm(x)

        x = self.ffn(x) + x
        x = self.out_norm(x)

        return x
