import re
import numpy as np
import torch
import chess, chess.pgn
from datasets import load_dataset, VerificationMode

from .position_datasets import PositionDataset, VariationNode


def san_to_uci(movetext: str, start_fen: str = chess.STARTING_FEN, chess960: bool = False) -> list[str]:
    """Convert a SAN movetext string (e.g. '1. e4 e5 2. Nf3 ...') to a list of UCI moves."""
    movetext = re.sub(r'\{[^}]*\}', '', movetext)
    board = chess.Board(start_fen, chess960=chess960)
    uci_moves = []
    for token in movetext.split():
        if token.endswith('.') or token in ('1-0', '0-1', '1/2-1/2', '*'):
            continue
        token = token.rstrip('?!')
        move = board.parse_san(token)
        uci_moves.append(move.uci())
        board.push(move)
    return uci_moves


class GamePositionDataset(PositionDataset):
    def __init__(self, forecast_depth: str, min_moves: int):
        super().__init__(forecast_depth)

        self.dataset = ...
        self.len = ...

        self.min_moves = min_moves
        self.forecast_depth = forecast_depth

        self.ignore_index = -100

    def __len__(self):
        return self.len


class LC0GamesDataset(GamePositionDataset):
    def __init__(self, forecast_depth, min_moves, split='train'):
        super().__init__(forecast_depth, min_moves)
        print('Preparing LC0GamesDataset')

        data_files = [
            'data/train-00049-of-00050.parquet'
        ]

        self.dataset = load_dataset(
            'groloch/lc0_games',
            split='train',
            data_files=data_files,
            verification_mode=VerificationMode.NO_CHECKS
        )
        if split == 'train':
            indices = range(500_000)
        elif split == 'val':
            indices = range(2_000_000, 2_010_000)
        elif split == 'test':
            indices = range(2_010_000, 2_020_000)

        self.dataset = self.dataset.filter(
            lambda x: x['variant'] == 'normal', num_proc=16
        ).map(
            lambda x: {
                'white': x['white'],
                'black': x['black'],
                'moves': x['moves'].split(' '),
                'start_fen': x['start_fen'],
                'game_length': len(x['moves'].split(' '))-min_moves
            },
            num_proc=16
        ).filter(
            lambda x: x['game_length'] > 0, num_proc=16
        ).select(indices)

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
        start_fen = game['start_fen'] if game['start_fen'] != '' else chess.STARTING_FEN
        board = chess.Board(start_fen)

        for k in range(self.min_moves+move_idx):
            board.push(chess.Move.from_uci(moves[k]))

        uci_move = moves[self.min_moves+move_idx] if self.min_moves+move_idx < len(moves) else None
        cp = None
        mate = None

        node = VariationNode(uci_move, cp=cp, mate=mate, expected_result=None)

        if self.min_moves+move_idx < len(moves):
            moves_ = moves[self.min_moves+move_idx:]
            # copy is needed since _get_forecast modifies the board state
            forecast, forecast_mask = self._get_forecast(board.copy(), moves_)
            forecast = forecast.reshape(64 * self.forecast_depth)
            forecast_mask = forecast_mask.reshape(64 * self.forecast_depth)
        else:
            forecast = None
            forecast_mask = None

        return self._process_item(board, [node], forecast, forecast_mask)
    

class LichessStandardGamesDataset(GamePositionDataset):
    def __init__(self, forecast_depth, min_moves):
        super().__init__(forecast_depth, min_moves)
        data_files = [
            'data/year=2025/month=01/train-00000-of-00072.parquet',
            'data/year=2025/month=01/train-00001-of-00072.parquet',
            'data/year=2025/month=01/train-00002-of-00072.parquet',
            'data/year=2025/month=01/train-00003-of-00072.parquet',
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
    def __init__(self, pgn, forecast_depth):
        super().__init__(forecast_depth)

        with open(pgn) as f:
            game = chess.pgn.read_game(f)
        moves = [move.uci() for move in game.mainline_moves()]
        assert len(moves) > 0, "Moves list cannot be empty"

        self.board = game.board()
        self.moves = moves
        self.forecast_depth = forecast_depth

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        # get uci move from san notation from moves list
        uci_move = self.board.parse_san(self.moves[0]).uci()
        node = VariationNode(uci_move, cp=None, mate=None, expected_result=None)

        forecast, forecast_mask = self._get_forecast(self.board.copy(), self.moves)

        return self._process_item(self.board, [node], forecast, forecast_mask)
