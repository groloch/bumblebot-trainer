import torch
from ...utils import ChessConstants


class SSLCollateFn:
    def __init__(self, max_lookahead: int):
        self.max_lookahead = max_lookahead

    def __call__(self, batch):
        tokens, tokens_, legal_moves, attacks, moves_list, lengths = zip(*batch)

        tokens = torch.stack(tokens)
        tokens_ = torch.stack(tokens_)

        legal_moves = torch.stack(legal_moves)
        attacks = torch.stack(attacks)

        batch_size = len(batch)
        max_len = 1 + self.max_lookahead

        moves = torch.full((batch_size, max_len), ChessConstants.NUM_POLICY_CLASSES, dtype=torch.long)
        for i, move_seq in enumerate(moves_list):
            moves[i, :len(move_seq)] = move_seq

        lengths = torch.tensor(lengths, dtype=torch.long)
        moves_attention_mask = (torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)).long()

        return tokens, tokens_, legal_moves, attacks, moves, moves_attention_mask
