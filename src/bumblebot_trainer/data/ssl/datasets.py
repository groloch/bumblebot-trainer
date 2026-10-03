import chess
import numpy as np

from ..game_datasets import LichessStandardGamesDataset
from ..iterable_datasets import LichessStandardIterableDataset
from .utils import encode_both_boards
from ..utils import san_to_uci, parse_result


def sample_ssl_future_indices(game_length: int, max_prediction_depth: int) -> tuple[int, int]:
    move_idx = np.random.randint(0, game_length - 1)

    max_lookahead = min(max_prediction_depth, game_length - move_idx)
    max_even_lookahead = max_lookahead - (max_lookahead % 2)

    lookahead = 2 * np.random.randint(1, max_even_lookahead // 2 + 1)
    return move_idx, move_idx + lookahead


class LichessStandardGamesSSLDataset(LichessStandardGamesDataset):
    """Dataset used for the SSL pipeline. It provides board encoding (now and future),
    the move sequence between the two boards, and the training targets for both boards:
    policy (next-move distribution) and value (TD-smoothed expected result).
    """
    def __init__(self, min_moves: int, max_prediction_depth: int, encoding: str):
        super().__init__(min_moves, encoding)

        self.dataset = self.dataset.filter(
            lambda x: x['game_length'] > max_prediction_depth, num_proc=16
        )
        self.len = len(self.dataset)

        self.max_prediction_depth = max_prediction_depth

    def __getitem__(self, idx):
        game = self.dataset[idx]
        game_length = game['game_length']

        move_idx, target_idx = sample_ssl_future_indices(
            game_length=game_length,
            max_prediction_depth=self.max_prediction_depth,
        )

        movelist = game['moves']
        result = game['result']
        board = chess.Board(chess960=True)

        for k in range(self.min_moves+move_idx):
            board.push(chess.Move.from_uci(movelist[k]))

        return encode_both_boards(
            board=board,
            encoding=self.encoding,
            min_moves=self.min_moves,
            move_idx=move_idx,
            target_idx=target_idx,
            movelist=movelist,
            result=result
        )


class LichessStandardIterableSSLDataset(LichessStandardIterableDataset):
    """Iterable dataset used for the SSL pipeline. It provides board encoding (now and future),
    the move sequence between the two boards, and the training targets for both boards:
    policy (next-move distribution) and value (TD-smoothed expected result).
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

            if game_length < 2:
                continue

            move_idx, target_idx = sample_ssl_future_indices(
                game_length=game_length,
                max_prediction_depth=self.max_prediction_depth,
            )

            board = chess.Board(chess960=True)

            for k in range(self.min_moves+move_idx):
                board.push(chess.Move.from_uci(moves[k]))

            yield encode_both_boards(
                board=board,
                encoding=self.encoding,
                min_moves=self.min_moves,
                move_idx=move_idx,
                target_idx=target_idx,
                movelist=moves,
                result=parse_result(item['Result'])
            )
