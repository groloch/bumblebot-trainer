import torch
from ...utils import ChessConstants

def ssl_collate_fn(batch):
    tokens, hm, ep_square, tokens_, hm_, ep_square_, legal_moves, moves_list, lengths = zip(*batch)

    tokens = torch.stack(tokens)
    hm = torch.stack(hm)
    ep_square = torch.stack(ep_square)
    tokens_ = torch.stack(tokens_)
    hm_ = torch.stack(hm_)
    ep_square_ = torch.stack(ep_square_)

    legal_moves = torch.stack(legal_moves)

    moves = torch.nn.utils.rnn.pad_sequence(
        moves_list, batch_first=True, padding_value=ChessConstants.NUM_POLICY_CLASSES
    )
    lengths = torch.tensor(lengths, dtype=torch.long)
    moves_attention_mask = (torch.arange(moves.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)).long()

    return tokens, hm, ep_square, tokens_, hm_, ep_square_, legal_moves, moves, moves_attention_mask
