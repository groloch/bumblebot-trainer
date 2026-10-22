import chess
import numpy as np
from torch.utils.data import Dataset, IterableDataset

from ..game_datasets import (
    LichessPuzzlesDataset,
    LichessStandardGamesDataset,
    T91GamesDataset,
)
from ..iterable_datasets import (
    LichessStandardIterableDataset,
    T91GamesIterableDataset,
)
from ..position_datasets import LichessEvalDataset
from .utils import encode_both_boards
from ..utils import VariationNode, san_to_uci


class MixedSSLDataset(Dataset):
    """Sample from several map-style SSL datasets according to their configured weights."""
    def __init__(self, datasets: list[Dataset], weights: list[float]):
        if len(datasets) == 0 or len(datasets) != len(weights):
            raise ValueError('datasets and weights must have the same non-zero length')
        if any(weight <= 0 for weight in weights):
            raise ValueError('dataset mix weights must be positive')

        self.datasets = datasets
        self.weights = np.asarray(weights, dtype=np.float64)
        self.weights /= self.weights.sum()

    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets) * 10

    def __getitem__(self, idx):
        dataset_idx = int(np.random.choice(len(self.datasets), p=self.weights))
        dataset = self.datasets[dataset_idx]
        sample_idx = idx % len(dataset)
        return dataset[sample_idx]


class MixedSSLIterableDataset(IterableDataset):
    """Interleave iterable SSL datasets according to their configured weights."""
    def __init__(self, datasets: list[IterableDataset], weights: list[float]):
        super().__init__()
        if len(datasets) == 0 or len(datasets) != len(weights):
            raise ValueError('datasets and weights must have the same non-zero length')
        if any(weight <= 0 for weight in weights):
            raise ValueError('dataset mix weights must be positive')

        self.datasets = datasets
        self.weights = np.asarray(weights, dtype=np.float64)

    def __iter__(self):
        active = list(range(len(self.datasets)))
        iterators = [iter(dataset) for dataset in self.datasets]

        while active:
            weights = self.weights[active]
            weights = weights / weights.sum()
            selected = int(np.random.choice(active, p=weights))
            try:
                yield next(iterators[selected])
            except StopIteration:
                active.remove(selected)


def sample_ssl_future_indices(game_length: int, max_prediction_depth: int) -> tuple[int, int]:
    if game_length < 2:
        raise ValueError('SSL examples require at least two available moves')
    if max_prediction_depth < 2:
        raise ValueError('max_prediction_depth must be at least two')

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
        super().__init__(0, encoding)

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

        for k in range(self.min_moves + move_idx):
            board.push(chess.Move.from_uci(movelist[k]))

        return encode_both_boards(
            board=board,
            nodes=[],
            nodes_=[],
            encoding=self.encoding,
            min_moves=self.min_moves,
            move_idx=move_idx,
            target_idx=target_idx,
            movelist=movelist,
            policy_valid=False,
            value_valid=False,
        )


class LichessStandardIterableSSLDataset(LichessStandardIterableDataset):
    """Iterable (streamed) dataset of human games used for the SSL pipeline only. It provides two positions
    from one game and the move sequence between the two. This dataset does not provide policy and value targets.
    """
    def __init__(self, min_moves: int, max_prediction_depth: int, encoding: str, min_elo: int):
        super().__init__(0, encoding, min_elo)
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

            for k in range(self.min_moves + move_idx):
                board.push(chess.Move.from_uci(moves[k]))

            yield encode_both_boards(
                board=board,
                nodes=[],
                nodes_=[],
                encoding=self.encoding,
                min_moves=self.min_moves,
                move_idx=move_idx,
                target_idx=target_idx,
                movelist=moves,
                policy_valid=False,
                value_valid=False,
            )


class LichessPuzzlesSSLDataset(LichessPuzzlesDataset):
    """Lichess puzzle positions with sharp policy and optional future-position targets."""
    def __init__(self, encoding: str, max_prediction_depth: int, split: str = 'train'):
        super().__init__(encoding, split)
        self.max_prediction_depth = max_prediction_depth

    def __getitem__(self, idx):
        puzzle_idx = np.searchsorted(self.puzzles_lengths, idx, side='right')
        if puzzle_idx == 0:
            move_idx = idx
        else:
            move_idx = idx - self.puzzles_lengths[puzzle_idx - 1]

        puzzle = self.dataset[puzzle_idx]
        moves = puzzle['moves']
        board = chess.Board(puzzle['fen'], chess960=True)
        for move in moves[:2 * move_idx + 1]:
            board.push_uci(move)

        move_start_idx = 2 * move_idx + 1
        nodes = [VariationNode(move=moves[move_start_idx], probability=1.0)]
        available_plies = len(moves) - move_start_idx
        max_lookahead = min(self.max_prediction_depth, available_plies)
        max_even_lookahead = max_lookahead - (max_lookahead % 2)

        if max_even_lookahead >= 2:
            lookahead = 2 * np.random.randint(1, max_even_lookahead // 2 + 1)
            target_idx = move_start_idx + lookahead
            ssl_valid = True
        else:
            target_idx = move_start_idx
            ssl_valid = False

        return encode_both_boards(
            board=board,
            nodes=nodes,
            nodes_=[],
            encoding=self.encoding,
            min_moves=0,
            move_idx=move_start_idx,
            target_idx=target_idx,
            movelist=moves,
            policy_valid=True,
            value_valid=False,
            ssl_valid=ssl_valid,
        )


class LichessEvalSSLDataset(LichessEvalDataset):
    """Stockfish-evaluated positions with policy, qvalue and optional PV future targets."""
    def __init__(self, encoding: str, max_prediction_depth: int, split: str = 'train'):
        super().__init__(encoding, split)
        self.max_prediction_depth = max_prediction_depth

    def __getitem__(self, index):
        sample = self.dataset[index]
        board, nodes, qvalue, _ = self._get_position_data(sample, index)

        candidate_pvs = []
        for line in sample['lines']:
            if not line:
                continue
            moves = line.split()
            max_lookahead = min(self.max_prediction_depth, len(moves))
            max_even_lookahead = max_lookahead - (max_lookahead % 2)
            if max_even_lookahead < 2:
                continue

            pv_board = board.copy()
            try:
                for move in moves[:max_even_lookahead]:
                    pv_board.push_uci(move)
            except ValueError:
                continue
            candidate_pvs.append((moves, max_even_lookahead))

        if candidate_pvs:
            pv_moves, max_even_lookahead = candidate_pvs[np.random.randint(len(candidate_pvs))]
            lookahead = 2 * np.random.randint(1, max_even_lookahead // 2 + 1)
            target_idx = lookahead
            ssl_valid = True
        else:
            pv_moves = []
            target_idx = 0
            ssl_valid = False

        return encode_both_boards(
            board=board,
            nodes=nodes,
            nodes_=[],
            encoding=self.encoding,
            min_moves=0,
            move_idx=0,
            target_idx=target_idx,
            movelist=pv_moves,
            qvalue=qvalue,
            policy_valid=True,
            value_valid=False,
            qvalue_valid=True,
            ssl_valid=ssl_valid,
        )


class T91GamesSSLDataset(T91GamesDataset):
    """Dataset of lc0 self-play games used for the SSL pipeline only. It provides two positions from
    one game and the move sequence between the two, plus dense policy, WDL, and qvalue targets.
    """
    def __init__(self, min_moves: int, max_prediction_depth: int, encoding: str):
        super().__init__(min_moves, encoding)
        self.max_prediction_depth = max_prediction_depth

        self.dataset = self.dataset.filter(
            lambda x: x['game_length'] > max_prediction_depth,
            num_proc=28
        )
        self.len = len(self.dataset)

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
        value, qvalue = self._get_wdl_and_q(board.turn, game, position_idx)
        value_, qvalue_ = self._get_wdl_and_q(board_.turn, game, target_position_idx)

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
            value_=value_,
            qvalue=qvalue,
            qvalue_=qvalue_,
        )


class T91GamesIterableSSLDataset(T91GamesIterableDataset):
    """Iterable T91 SSL samples with dense policy, WDL, and qvalue targets.
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

                value = self._get_wdl(board.turn, item, position_idx)
                value_ = self._get_wdl(board_.turn, item, target_position_idx)
                qvalue = self._get_qvalue(board.turn, item, position_idx)
                qvalue_ = self._get_qvalue(board_.turn, item, target_position_idx)

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
                    value_=value_,
                    qvalue=qvalue,
                    qvalue_=qvalue_,
                )
