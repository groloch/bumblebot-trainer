import re
import numpy as np
import torch
import chess, chess.pgn
from datasets import load_dataset, VerificationMode, Value

from .position_datasets import PositionDataset
from .utils import san_to_uci, process_item, VariationNode, parse_result, encode_board, get_move_id


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
                'game_length': len(x['moves'])-min_moves,
                'result': parse_result(x['Result'])
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

        node = VariationNode(uci_move, cp=cp, mate=mate)

        return process_item(board, [node], self.encoding, self.temperature)


class Lc0GamesDataset(GamePositionDataset):
    """Lc0Games is a dataset of 170m games played by Leela Chess Zero against itself.
    The moves are written in uci format, and some games are chess960 games for which
    the start fen is specified. To keep the dataset to a manageable size it is limited
    to the first 4 shards here.

    This dataset is licensed under the odbl licence
    """
    def __init__(self, min_moves, encoding: str):
        super().__init__(min_moves, encoding)
        data_files = [
            'data/train-00000-of-00050.parquet',
            'data/train-00001-of-00050.parquet',
            # 'data/train-00002-of-00050.parquet',
            # 'data/train-00003-of-00050.parquet',

            # 'data/train-00004-of-00050.parquet',
            # 'data/train-00005-of-00050.parquet',
            # 'data/train-00006-of-00050.parquet',
            # 'data/train-00007-of-00050.parquet',
        ]

        self.dataset = load_dataset(
            'groloch/lc0_games',
            split='train',
            data_files=data_files,
            verification_mode=VerificationMode.NO_CHECKS
        )

        self.dataset = self.dataset.filter(
            lambda x: x['variant'] == 'normal'
        ).map(
            lambda x: {
                'moves': x['moves'].split(),
            },
            num_proc=16
        ).map(
            lambda x: {
                'moves': x['moves'],
                'game_length': len(x['moves'])-min_moves,
                'result': parse_result(x['result'])
            },
            num_proc=16
        ).filter(
            lambda x: x['game_length'] > 0, num_proc=16
        ).cast_column(
            'result', Value('int32')
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
        board = chess.Board(chess960=True)

        for k in range(self.min_moves+move_idx):
            board.push(chess.Move.from_uci(moves[k]))

        uci_move = moves[self.min_moves+move_idx] if self.min_moves+move_idx < len(moves) else None
        cp = None
        mate = None

        node = VariationNode(uci_move, cp=cp, mate=mate)

        return process_item(board, [node], self.encoding, self.temperature)


class T91GamesDataset(GamePositionDataset):
    def __init__(self, min_moves: int, encoding: str):
        super().__init__(min_moves, encoding)

        # self.dataset = load_dataset(
        #     'groloch/lc0_training_T91_10m',
        #     split='train',
        # )
        self.dataset = load_dataset(
            'parquet',
            data_files='/home/baptiste/data/ai/datasets/lc0_training/tmp/*.parquet',
            split='train',
        )

        self.dataset = self.dataset.map(
            lambda x: {**x, 'game_length': len(x['movelist'].split(' '))-min_moves},
            num_proc=28
        ).filter(
            lambda x: x['game_length'] > 0,
            num_proc=28
        )

        self.games_lengths = np.cumsum(self.dataset['game_length'])
        self.len = int(self.games_lengths[-1])

        self.wdl_temp = 0.15

    def _get_nodes(self, board: chess.Board, game, position_idx: int):
        legal_moves = sorted(move.uci() for move in board.legal_moves)
        policy = game['policies'][position_idx]

        if len(legal_moves) != len(policy):
            raise ValueError('Policy length does not match the number of legal moves')

        return [
            VariationNode(move=move, probability=percentage/100.0)
            for move, percentage in zip(legal_moves, policy)
        ]

    def _get_wdl(self, board: chess.Board, game, position_idx: int) -> float:
        value = game['qvalues'][position_idx]
        dvalue = game['dvalues'][position_idx]

        if value is None:
            i = 1
            while value is None:
                value = game['qvalues'][position_idx-i]
                i+=1
        if dvalue is None:
            i = 1
            while dvalue is None:
                dvalue = game['qvalues'][position_idx-i]
                i+=1

        if board.turn == chess.BLACK:
            value = 1.0 - value

        win = (value + 1) / 2
        draw = dvalue
        loss = 1-win

        wdl = torch.as_tensor([win, draw, loss], dtype=float)
        return (wdl / self.wdl_temp).softmax(dim=-1)

    def __getitem__(self, idx):
        game_idx = np.searchsorted(self.games_lengths, idx, side='right')
        if game_idx == 0:
            move_idx = idx
        else:
            move_idx = idx - self.games_lengths[game_idx - 1]

        game = self.dataset[game_idx]
        position_idx = self.min_moves + move_idx
        fen = game['fen'] if game['fen'] is not None else chess.STARTING_FEN
        board = chess.Board(fen, chess960=True)

        for move in game['moves'][:position_idx]:
            board.push_uci(move)

        nodes = self._get_nodes(board, game, position_idx)
        value = self._get_wdl(board, game, position_idx)
        return process_item(
            board,
            nodes,
            self.encoding,
            self.temperature,
            value=value
        )


class LichessPuzzlesDataset(PositionDataset):
    """Dataset of Lichess puzzles, flattened over all of their solution moves.

    Each puzzle is a FEN plus a move list that starts with the opponent's setup
    move, followed by the player's solution moves interleaved with forced
    opponent replies. The player's moves are therefore at odd indices
    (1, 3, 5, ...) and the list always ends on a player move.

    The dataset exposes one position per player move: item ``idx`` is the
    position in which the player has to find their ``move_idx``-th solution move.
    """
    def __init__(self, encoding, num_puzzles: int = 100_000):
        super().__init__(encoding)

        self.dataset = load_dataset('Lichess/chess-puzzles', split='train')
        num_puzzles = min(num_puzzles, len(self.dataset))

        selected_themes = (
            # Move type
            'defensiveMove', 'zugzwang', 'sacrifice', 'mate', 'quietMove', 'fork',
            'skewer', 'pin', 'intermezzo', 'clearance', 'attraction', 'xRayAttack',
            'advancedPawn',

            # Game phase
            'opening', 'middlegame', 'endgame',
            ''
        )

        self.dataset = self.dataset.select(range(num_puzzles)).map(
            lambda x: {
                'fen': x['FEN'],
                'moves': x['Moves'].split(' '),
                'rating': x['Rating'],
                'themes': [t for t in x['Themes'] if t in selected_themes],
                'puzzle_len': len(x['Moves'].split(' ')) // 2
            }, num_proc=16
        )

        self.puzzles_lengths = np.cumsum(self.dataset['puzzle_len'])
        # number of solution (player) moves per puzzle, indexed by puzzle index
        self.puzzle_lens = np.diff(self.puzzles_lengths, prepend=0).astype(np.int64)
        self.num_puzzles = len(self.dataset)
        self.len = int(self.puzzles_lengths[-1])

    def __len__(self):
        return int(self.len)

    def __getitem__(self, idx):
        puzzle_idx = np.searchsorted(self.puzzles_lengths, idx, side='right')
        if puzzle_idx == 0:
            move_idx = idx
        else:
            move_idx = idx - self.puzzles_lengths[puzzle_idx - 1]

        puzzle = self.dataset[puzzle_idx]
        moves = puzzle['moves']
        board = chess.Board(puzzle['fen'], chess960=True)

        for k in range(2 * move_idx + 1):
            board.push_uci(moves[k])

        uci_move = moves[2 * move_idx + 1]

        board_, extra_ = encode_board(board, self.encoding)
        move_ = torch.tensor(
            get_move_id(chess.Move.from_uci(uci_move), board.turn), dtype=torch.long
        )
        puzzle_idx_ = torch.tensor(puzzle_idx, dtype=torch.long)
        rating_ = torch.tensor(float(puzzle['rating']), dtype=torch.float32)

        return board_, move_, puzzle_idx_, rating_, extra_


class SingleGameDataset(PositionDataset):
    """For testing purposes, a dataset made of a single game
    """
    def __init__(self, pgn, encoding: str):
        super().__init__(min_moves=0, encoding=encoding)

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

        return process_item(self.board, [node], self.encoding, self.temperature)
