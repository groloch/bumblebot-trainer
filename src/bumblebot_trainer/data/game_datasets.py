import re
import numpy as np
import torch
import chess, chess.pgn
from datasets import load_dataset, VerificationMode

from .position_datasets import PositionDataset, VariationNode
from .utils import san_to_uci


class GamePositionDataset(PositionDataset):
    """Abstract class for datasets of positions extracted from games."""
    def __init__(self, min_moves: int, encoding: str):
        super().__init__(encoding)

        self.dataset = ...
        self.len = ...

        self.min_moves = min_moves

        self.ignore_index = -100

    def __len__(self):
        return self.len


class LichessStandardGamesDataset(GamePositionDataset):
    """LichessStandardGames is a billion-game-scale dataset that contains rated human games played
    on lichess from 2013 to current date. Note: some of the games contain stockfish evaluations
    (from various versions of stockfish and depth), which we filter out here.

    This dataset is licensed under the cc0-1.0 licence
    """
    def __init__(self, min_moves, encoding: str):
        super().__init__(min_moves, encoding)
        data_files = [
            'data/year=2025/month=01/train-00000-of-00072.parquet',
            'data/year=2025/month=01/train-00001-of-00072.parquet',
            'data/year=2025/month=01/train-00002-of-00072.parquet',
            'data/year=2025/month=01/train-00003-of-00072.parquet',

            # 'data/year=2025/month=01/train-00004-of-00072.parquet',
            # 'data/year=2025/month=01/train-00005-of-00072.parquet',
            # 'data/year=2025/month=01/train-00006-of-00072.parquet',
            # 'data/year=2025/month=01/train-00007-of-00072.parquet',
            # 'data/year=2025/month=01/train-00008-of-00072.parquet',
            # 'data/year=2025/month=01/train-00009-of-00072.parquet',
            # 'data/year=2025/month=01/train-00010-of-00072.parquet',
        ]

        self.dataset = load_dataset(
            'Lichess/standard-chess-games',
            split='train',
            data_files=data_files
        )

        self.dataset = self.dataset.map(
            lambda x: {
                'moves': san_to_uci(x['movetext']),
            },
            num_proc=16
        ).map(
            lambda x: {
                'moves': x['moves'],
                'game_length': len(x['moves'])-min_moves
            },
            num_proc=16
        ).filter(
            lambda x: x['game_length'] > 0, num_proc=16
        )

        self.games_lengths = np.cumsum(self.dataset['game_length'])
        self.len = self.games_lengths[-1]

    def __getitem__(self, idx):
        game_idx = np.searchsorted(self.games_lengths, idx, side='right')
        if game_idx == 0:
            move_idx = idx
        else:
            move_idx = idx - self.games_lengths[game_idx - 1]

        game = self.dataset[game_idx]
        moves = game['moves']
        board = chess.Board()

        for k in range(self.min_moves+move_idx):
            board.push(chess.Move.from_uci(moves[k]))

        uci_move = moves[self.min_moves+move_idx] if self.min_moves+move_idx < len(moves) else None
        cp = None
        mate = None

        node = VariationNode(uci_move, cp=cp, mate=mate, expected_result=None)

        return self._process_item(board, [node])


class SingleGameDataset(PositionDataset):
    """For testing purposes, a dataset made of a single game
    """
    def __init__(self, pgn, encoding: str):
        super().__init__(encoding)

        with open(pgn) as f:
            game = chess.pgn.read_game(f)
        moves = [move.uci() for move in game.mainline_moves()]
        assert len(moves) > 0, "Moves list cannot be empty"

        self.board = game.board()
        self.moves = moves

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        # get uci move from san notation from moves list
        uci_move = self.board.parse_san(self.moves[0]).uci()
        node = VariationNode(uci_move, cp=None, mate=None, expected_result=None)

        return self._process_item(self.board, [node])
