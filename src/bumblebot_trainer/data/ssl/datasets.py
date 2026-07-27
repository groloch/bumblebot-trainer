import torch
import chess
import numpy as np

from ..game_datasets import LichessStandardGamesDataset
from ..iterable_datasets import LichessStandardIterableDataset
from .utils import (
    encode_move_for_predictor,
    SSLConstants,
    get_relative_attack_map,
    encode_both_boards
)
from ..utils import encode_board, san_to_uci
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
        board = chess.Board(chess960=True)

        for k in range(self.min_moves+move_idx):
            board.push(chess.Move.from_uci(movelist[k]))

        return encode_both_boards(
            board=board,
            encoding=self.encoding,
            min_moves=self.min_moves,
            move_idx=move_idx,
            target_idx=target_idx,
            movelist=movelist
        )


class LichessStandardIterableSSLDataset(LichessStandardIterableDataset):
    """Iterable dataset used for the SSL pipeline. It provides board encoding (now and future),
    the move sequence between the two boards, and the square level targets: legal moves and
    attack maps for both boards.
    This dataset streams the dataset to filter out low-elo games without caching the entire
    dataset in memory.
    """
    def __init__(self, min_moves: int, max_prediction_depth: int, encoding: str, min_elo: int):
        super().__init__(min_moves, encoding, min_elo)
        self.max_prediction_depth = max_prediction_depth

    def __iter__(self):
        for item in self.dataset:
            if item['WhiteElo'] < self.min_elo or item['BlackElo'] < self.min_elo:
                continue

            moves = san_to_uci(item['movetext'])
            game_length = len(moves) - self.min_moves

            if game_length <= 0:
                continue

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

            board = chess.Board(chess960=True)

            for k in range(self.min_moves+move_idx):
                board.push(chess.Move.from_uci(moves[k]))

            yield encode_both_boards(
                board=board,
                encoding=self.encoding,
                min_moves=self.min_moves,
                move_idx=move_idx,
                target_idx=target_idx,
                movelist=moves
            )
