import torch
import chess
import numpy as np

from ..game_datasets import LichessStandardGamesDataset
from ...utils import encode_board, get_move_id, ChessConstants


class LichessStandardGamesSSLDataset(LichessStandardGamesDataset):
    def __init__(self, min_moves: int, max_prediction_depth: int):
        super().__init__(min_moves)

        self.dataset = self.dataset.filter(
            lambda x: x['game_length'] > max_prediction_depth, num_proc=16
        )
        self.len  = len(self.dataset)

        self.max_prediction_depth = max_prediction_depth

    def __getitem__(self, idx):
        game_length = self.dataset[idx]['game_length']
        move_idx = np.random.randint(0, game_length)
        max_target_idx = min(move_idx + self.max_prediction_depth, game_length)
        target_idx = np.random.randint(
            move_idx + 1,
            max_target_idx + 1,
        )

        game = self.dataset[idx]
        movelist = game['moves']
        board = chess.Board()

        for k in range(self.min_moves+move_idx):
            board.push(chess.Move.from_uci(movelist[k]))

        legal_moves = torch.zeros(ChessConstants.NUM_POLICY_CLASSES, dtype=torch.bool)
        for move in board.legal_moves:
            legal_moves[get_move_id(move, board.turn)] = True


        tokens, hm, ep_square = encode_board(board)
        movelist_ = torch.zeros(target_idx - move_idx, dtype=torch.long)

        for k in range(self.min_moves+move_idx, self.min_moves+target_idx):
            move = chess.Move.from_uci(movelist[k])
            movelist_[k - (self.min_moves+move_idx)] = get_move_id(move, board.turn)
            board.push(move)

        tokens_, hm_, ep_square_ = encode_board(board)

        return tokens, hm, ep_square, tokens_, hm_, ep_square_, legal_moves, movelist_, target_idx - move_idx


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
