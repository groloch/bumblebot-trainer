import torch
import chess
import numpy as np

from ..game_datasets import LichessStandardGamesDataset
from .utils import encode_move_for_predictor, SSLConstants
from ..utils import encode_board
from ...utils import get_move_id, ChessConstants


class LichessStandardGamesSSLDataset(LichessStandardGamesDataset):
    """Dataset used for the SSL pipeline. It provides board encoding (now and future),
    the move sequence between the two boards, and the square level targets: legal moves and
    attack maps for both boards.
    """
    def __init__(self, min_moves: int, max_prediction_depth: int, encoding: str):
        super().__init__(min_moves, encoding)

        self.dataset = self.dataset.filter(
            lambda x: x['game_length'] > max_prediction_depth, num_proc=16
        )
        self.len  = len(self.dataset)

        self.max_prediction_depth = max_prediction_depth

    def _get_relative_attack_map(self, board: chess.Board):
        """Relative attack map: number of attackers (control) we have
        over a square minus number of attackers they have.
        """
        if board.turn == chess.BLACK:
            board = board.mirror()

        attack_map = torch.zeros(64)
        for square in chess.SQUARES:
            attack_map[square] += len(board.attackers(chess.WHITE, square))
            attack_map[square] -= len(board.attackers(chess.BLACK, square))
        attack_map += ChessConstants.MAX_ATTACKERS
        attack_map.clamp_(0, ChessConstants.MAX_ATTACKERS * 2)
        return attack_map.long()

    def __getitem__(self, idx):
        game_length = self.dataset[idx]['game_length']
        move_idx = np.random.randint(0, game_length)
        max_target_idx = min(move_idx + self.max_prediction_depth, game_length)
        target_idx = np.random.randint(
            move_idx,
            max_target_idx + 1,
        )
        if (target_idx - move_idx) % 2 != 0:
            if target_idx == max_target_idx:
                target_idx -= 1
            else:
                target_idx += 1

        game = self.dataset[idx]
        movelist = game['moves']
        board = chess.Board()

        for k in range(self.min_moves+move_idx):
            board.push(chess.Move.from_uci(movelist[k]))

        legal_moves = torch.zeros(ChessConstants.NUM_POLICY_CLASSES, dtype=torch.bool)
        for move in board.legal_moves:
            legal_moves[get_move_id(move, board.turn)] = True

        tokens = encode_board(board, self.encoding)
        attacks = self._get_relative_attack_map(board)
        movelist_ = torch.zeros((target_idx - move_idx, SSLConstants.NUM_TOKENS_PER_MOVE), dtype=torch.long)

        _board = board.copy(stack=False)
        for k in range(self.min_moves+move_idx, self.min_moves+target_idx):
            move = chess.Move.from_uci(movelist[k])

            moved_piece_type = _board.piece_at(move.from_square).piece_type

            if _board.piece_at(move.to_square) is None:
                if move.to_square == _board.ep_square and moved_piece_type == chess.PAWN:
                    taken_piece_type = chess.PAWN
                else:
                    taken_piece_type = None
            else:
                if _board.piece_at(move.to_square).color == _board.turn:
                    taken_piece_type = None
                else:
                    taken_piece_type = _board.piece_at(move.to_square).piece_type

            movelist_[k - (self.min_moves+move_idx), :] = encode_move_for_predictor(
                move=move,
                piece_type=moved_piece_type,
                taken_piece_type=taken_piece_type,
                turn=_board.turn,
                perspective=board.turn
            )
            _board.push(move)

        tokens_ = encode_board(_board, self.encoding)
        attacks_ = self._get_relative_attack_map(_board)
        legal_moves_ = torch.zeros(ChessConstants.NUM_POLICY_CLASSES, dtype=torch.bool)
        for move in _board.legal_moves:
            legal_moves_[get_move_id(move, _board.turn)] = True

        return tokens, tokens_, legal_moves, attacks, legal_moves_, attacks_, movelist_, target_idx - move_idx
