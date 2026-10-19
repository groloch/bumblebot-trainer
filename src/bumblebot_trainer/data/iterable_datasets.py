from torch.utils.data import IterableDataset
from datasets import load_dataset
import chess
import numpy as np

from .utils import (
    process_item,
    VariationNode,
    san_to_uci,
    get_lichess_files_for_year,
    WdlDataset
)


class IterablePositionDataset(IterableDataset):
    def __init__(self, encoding: str):
        self.encoding = encoding
        self.temperature = 0.1

    def __iter__(self):
        raise NotImplementedError()


class LichessStandardIterableDataset(IterablePositionDataset):
    """LichessStandardGames is a billion-game-scale dataset that contains rated human games played
    on lichess from 2013 to current date. Note: some of the games contain stockfish evaluations
    (from various versions of stockfish and depth), which we filter out here.

    This dataset is licensed under the cc0-1.0 licence
    """
    def __init__(self, min_moves: int, encoding: str, min_elo: int):
        super().__init__(encoding)

        self.min_moves = min_moves
        self.min_elo = min_elo

        data_files = get_lichess_files_for_year(2025)
        data_files += get_lichess_files_for_year(2024)
        data_files += get_lichess_files_for_year(2023)

        self.dataset = load_dataset(
            'Lichess/standard-chess-games',
            split='train',
            data_files=data_files,
            streaming=True
        )

    def __iter__(self):
        for item in self.dataset:
            if item['WhiteElo'] < self.min_elo or item['BlackElo'] < self.min_elo:
                continue

            moves = san_to_uci(item['movetext'])
            game_length = len(moves) - self.min_moves

            if game_length <= 0:
                continue

            board = chess.Board()

            for move_idx in range(game_length):
                for k in range(self.min_moves+move_idx):
                    board.push(chess.Move.from_uci(moves[k]))

                uci_move = moves[self.min_moves+move_idx] if self.min_moves+move_idx < len(moves) else None
                cp = None
                mate = None

                node = VariationNode(uci_move, cp=cp, mate=mate)

                yield process_item(board, [node], self.encoding, self.temperature)


class T91GamesIterableDataset(IterablePositionDataset, WdlDataset):
    """groloch/lc0_training_T91_10m is a dataset of 10m games played Leela Chess Zero
     against itself during its extended RL runs.

    This dataset is licensed under the odbl licence
    """
    def __init__(
            self,
            min_moves: int,
            encoding: str,
            shuffle_buffer_size: int = 10_000,
            seed: int = 0
        ):
        super().__init__(encoding)
        WdlDataset.__init__(self)

        self.min_moves = min_moves

        # self.dataset = load_dataset(
        #     'groloch/lc0_training_T91_10m',
        #     split='train',
        #     streaming=True
        # )

        self.dataset = load_dataset(
            'parquet',
            data_files='/home/baptiste/data/ai/datasets/lc0_training/tmp/*.parquet',
            split='train',
            streaming=True
        )

        self.dataset = self.dataset.shuffle(seed=seed, buffer_size=shuffle_buffer_size)

    def __iter__(self):
        while True:
            for item in self.dataset:
                moves = item['movelist'].split()
                game_length = len(moves) - self.min_moves

                if game_length <= 0:
                    continue

                move_idx = self.min_moves + np.random.randint(game_length)
                fen = item['fen'] if item['fen'] is not None else chess.STARTING_FEN
                board = chess.Board(fen, chess960=True)

                for move in moves[:move_idx]:
                    board.push_uci(move)

                legal_moves = sorted(move.uci() for move in board.legal_moves)
                policy = item['policies'][move_idx]

                if len(legal_moves) != len(policy):
                    raise ValueError('T91 policy length does not match the number of legal moves')

                nodes = self._get_nodes(board, item, move_idx)
                value = self._get_wdl(board, item, move_idx)
                qvalue = self._get_qvalue(board.turn, item, move_idx)
                yield process_item(
                    board,
                    nodes,
                    self.encoding,
                    self.temperature,
                    value=value,
                    qvalue=qvalue,
                )
