import chess
import numpy as np

from ..game_datasets import LichessStandardGamesDataset, T91GamesDataset
from ..iterable_datasets import (
    LichessStandardIterableDataset,
    T91GamesIterableDataset,
)
from .utils import encode_both_boards
from ..utils import VariationNode, san_to_uci


def sample_ssl_future_indices(game_length: int, max_prediction_depth: int) -> tuple[int, int]:
    move_idx = np.random.randint(0, game_length - 1)

    max_lookahead = min(max_prediction_depth, game_length - move_idx)
    max_even_lookahead = max_lookahead - (max_lookahead % 2)

    lookahead = 2 * np.random.randint(1, max_even_lookahead // 2 + 1)
    return move_idx, move_idx + lookahead


class LichessStandardGamesSSLDataset(LichessStandardGamesDataset):
    """Dataset of human games used for the SSL pipeline only. It provides two positions from one game
    and the move sequence between the two. This dataset does not provide policy and value targets.
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
        board = chess.Board(chess960=True)

        for k in range(self.min_moves+move_idx):
            board.push(chess.Move.from_uci(movelist[k]))

        # TODO fill


class LichessStandardIterableSSLDataset(LichessStandardIterableDataset):
    """Iterable (streamed) dataset of human games used for the SSL pipeline only. It provides two positions
    from one game and the move sequence between the two. This dataset does not provide policy and value targets.
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

            # TODO fill


class T91GamesSSLDataset(T91GamesDataset):
    """Dataset of lc0 self-play games used for the SSL pipeline only. It provides two positions from
    one game and the move sequence between the two. This dataset also provides dense policy and wdl/scalar value
    targets.
    """
    def __init__(self, min_moves: int, max_prediction_depth: int, encoding: str):
        super().__init__(min_moves, encoding)
        self.max_prediction_depth = max_prediction_depth

        self.dataset = self.dataset.filter(
            lambda x: x['game_length'] > max_prediction_depth,
            num_proc=16
        )
        self.len = len(self.dataset)

    def __len__(self):
        return self.len * 1000

    def __getitem__(self, idx):
        idx = idx % self.len

        game = self.dataset[idx]
        game_length = game['game_length']
        move_idx, target_idx = sample_ssl_future_indices(
            game_length=game_length - 1,
            max_prediction_depth=self.max_prediction_depth,
        )
        moves = game['movelist'].split(' ')

        position_idx = self.min_moves + move_idx
        target_position_idx = self.min_moves + target_idx
        fen = game['fen'] if game['fen'] is not None else chess.STARTING_FEN
        board = chess.Board(fen, chess960=True)
        for move in moves[:position_idx]:
            board.push_uci(move)

        board_ = board.copy()
        for move in moves[position_idx:target_position_idx]:
            board_.push_uci(move)

        nodes = self._get_nodes(board, game, position_idx)
        nodes_ = self._get_nodes(board_, game, target_position_idx)
        value = self._get_wdl(board, game, position_idx)
        value_ = self._get_wdl(board_, game, target_position_idx)

        return encode_both_boards(
            board=board,
            nodes=nodes,
            nodes_=nodes_,
            encoding=self.encoding,
            min_moves=self.min_moves,
            move_idx=move_idx,
            target_idx=target_idx,
            movelist=moves,
            value=value,
            value_=value_
        )


class T91GamesIterableSSLDataset(T91GamesIterableDataset):
    """Iterable (streamed) dataset of lc0 self-play games used for the SSL pipeline only. It provides two
    positions from one game and the move sequence between the two. This dataset also provides dense policy
    and wdl/scalar value targets.
    """
    def __init__(
            self,
            min_moves: int,
            max_prediction_depth: int,
            encoding: str,
            shuffle_buffer_size: int = 10_000,
            seed: int = 0
        ):
        super().__init__(min_moves, encoding, shuffle_buffer_size, seed)
        self.max_prediction_depth = max_prediction_depth

    def _policy_nodes(self, board, policy):
        legal_moves = sorted([move.uci() for move in board.legal_moves])
        return [
            VariationNode(move=move, probability=percentage/100.0)
            for move, percentage in zip(legal_moves, policy)
        ]

    def __iter__(self):
        while True:
            for item in self.dataset:
                moves = item['movelist'].split()
                available_positions = len(moves) - self.min_moves

                if available_positions < 3:
                    continue

                move_idx, target_idx = sample_ssl_future_indices(
                    game_length=available_positions - 1,
                    max_prediction_depth=self.max_prediction_depth,
                )

                position_idx = self.min_moves + move_idx
                target_position_idx = self.min_moves + target_idx
                fen = item['fen'] if item['fen'] is not None else chess.STARTING_FEN
                board = chess.Board(fen, chess960=True)
                for move in moves[:position_idx]:
                    board.push_uci(move)

                board_ = board.copy()
                for move in moves[position_idx:target_position_idx]:
                    board_.push_uci(move)

                value = min(max(float(item['qvalues'][position_idx]), 0.0), 1.0)
                if board.turn == chess.BLACK:
                    value = 1.0 - value

                value_ = min(max(float(item['qvalues'][target_position_idx]), 0.0), 1.0)
                if board_.turn == chess.BLACK:
                    value_ = 1.0 - value_

                yield encode_both_boards(
                    board=board,
                    nodes=self._policy_nodes(board, item['policies'][position_idx]),
                    nodes_=self._policy_nodes(board_, item['policies'][target_position_idx]),
                    encoding=self.encoding,
                    min_moves=self.min_moves,
                    move_idx=move_idx,
                    target_idx=target_idx,
                    movelist=moves,
                    value=value,
                    value_=value_
                )
