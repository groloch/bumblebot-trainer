import chess
import numpy as np

from ..game_datasets import LichessStandardGamesDataset, Lc0GamesDataset, T91GamesDataset
from ..iterable_datasets import (
    LichessStandardIterableDataset,
    Lc0GamesIterableDataset,
    T91GamesIterableDataset,
)
from .utils import encode_both_boards, apply_td_value
from ..utils import VariationNode, san_to_uci, parse_result


def _make_result_targets(board, movelist, min_moves, move_idx, target_idx, result):
    position_idx = min_moves + move_idx
    target_position_idx = min_moves + target_idx

    value = apply_td_value(
        board.turn,
        position_idx / len(movelist),
        result
    )
    target_turn = board.turn if (target_idx - move_idx) % 2 == 0 else not board.turn
    value_ = apply_td_value(
        target_turn,
        target_position_idx / len(movelist),
        result
    )

    nodes = [VariationNode(movelist[position_idx], expected_result=value)]
    nodes_ = None
    if target_position_idx < len(movelist):
        nodes_ = [VariationNode(movelist[target_position_idx], expected_result=value_)]

    return nodes, nodes_, value, value_


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

        nodes, nodes_, value, value_ = _make_result_targets(
            board, movelist, self.min_moves, move_idx, target_idx, result
        )
        return encode_both_boards(
            board=board,
            nodes=nodes,
            nodes_=nodes_,
            encoding=self.encoding,
            min_moves=self.min_moves,
            move_idx=move_idx,
            target_idx=target_idx,
            movelist=movelist,
            value=value,
            value_=value_
        )


class Lc0GamesSSLDataset(Lc0GamesDataset):
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
        board = chess.Board(game['start_fen'] or chess.STARTING_FEN, chess960=True)

        for k in range(self.min_moves+move_idx):
            board.push(chess.Move.from_uci(movelist[k]))

        nodes, nodes_, value, value_ = _make_result_targets(
            board, movelist, self.min_moves, move_idx, target_idx, result
        )
        return encode_both_boards(
            board=board,
            nodes=nodes,
            nodes_=nodes_,
            encoding=self.encoding,
            min_moves=self.min_moves,
            move_idx=move_idx,
            target_idx=target_idx,
            movelist=movelist,
            value=value,
            value_=value_
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

            result = parse_result(item['Result'])
            nodes, nodes_, value, value_ = _make_result_targets(
                board, moves, self.min_moves, move_idx, target_idx, result
            )
            yield encode_both_boards(
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


class Lc0GamesIterableSSLDataset(Lc0GamesIterableDataset):
    """Iterable dataset used for the SSL pipeline. It provides board encoding (now and future),
    the move sequence between the two boards, and the training targets for both boards:
    policy (next-move distribution) and value (TD-smoothed expected result).
    This dataset streams lc0 self-play games, shuffling them with a buffer to avoid caching
    the entire dataset in memory.
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

    def __iter__(self):
        for item in self.dataset:
            moves = item['moves'].split()
            game_length = len(moves) - self.min_moves

            if game_length < 2:
                continue

            move_idx, target_idx = sample_ssl_future_indices(
                game_length=game_length,
                max_prediction_depth=self.max_prediction_depth,
            )

            board = chess.Board(item['start_fen'] or chess.STARTING_FEN, chess960=True)

            for k in range(self.min_moves+move_idx):
                board.push(chess.Move.from_uci(moves[k]))

            result = parse_result(item['result'])
            nodes, nodes_, value, value_ = _make_result_targets(
                board, moves, self.min_moves, move_idx, target_idx, result
            )
            yield encode_both_boards(
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


class T91GamesSSLDataset(T91GamesDataset):
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
