import torch
import torch.nn as nn

from ..utils import ChessVocabulary, ChessConstants


class Embedding(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.embedding = nn.Embedding(ChessVocabulary.TOTAL_TOKENS, hidden_size)

        self.pos = nn.Parameter(torch.randn(ChessConstants.CONTEXT_LENGTH, hidden_size) * 0.02)
        self.epsq_bias = nn.Parameter(torch.randn(1, hidden_size) * 0.02)

        self.hm_embed = nn.Linear(1, 2*hidden_size)

    def forward(self, tokens, hm, epsq):
        x = self.embedding(tokens) + self.pos

        mask = (epsq != -1).to(x.dtype).view(-1, 1, 1)
        x = x + mask * self.epsq_bias

        a, m = self.hm_embed(hm.unsqueeze(-1)).chunk(2, dim=-1)
        x[:, -1, :] *= a
        x[:, -1, :] += m

        return x
