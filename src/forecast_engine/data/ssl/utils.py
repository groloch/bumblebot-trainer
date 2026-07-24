import torch
from ...utils import ChessConstants


def ssl_collate_fn(batch):
    tokens, tokens_, legal_moves, moves_list, lengths = zip(*batch)

    tokens = torch.stack(tokens)
    tokens_ = torch.stack(tokens_)

    legal_moves = torch.stack(legal_moves)

    moves = torch.nn.utils.rnn.pad_sequence(
        moves_list, batch_first=True, padding_value=ChessConstants.NUM_POLICY_CLASSES
    )
    lengths = torch.tensor(lengths, dtype=torch.long)
    moves_attention_mask = (torch.arange(moves.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)).long()

    return tokens, tokens_, legal_moves, moves, moves_attention_mask
