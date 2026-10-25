import re
import numpy as np
import torch
import chess, chess.pgn
from datasets import load_dataset

from .position_datasets import PositionDataset
from .utils import (
    san_to_uci,
    process_item,
    VariationNode,
    parse_result,
    encode_board,
    get_move_id,
    get_game_phase,
    WdlDataset
)


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
        print('Preparing LichessStandardGamesDataset')
        data_files = [
            'data/year=2025/month=01/train-00000-of-00072.parquet',
            # 'data/year=2025/month=01/train-00001-of-00072.parquet',
            # 'data/year=2025/month=01/train-00002-of-00072.parquet',
            # 'data/year=2025/month=01/train-00003-of-00072.parquet',

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


class T91GamesDataset(GamePositionDataset, WdlDataset):
    """This is a dataset containing a portion (~10m games; ~5% of the total to this date)
    of the test91 large-scale reinforcement training run self-play games. This dataset contains
    dense policy, WDL, and scalar qvalue targets.

    This dataset is licensed under the odbl license.
    """
    def __init__(
            self,
            min_moves: int,
            encoding: str,
            tag: str = 'all'):
        super().__init__(min_moves, encoding)
        WdlDataset.__init__(self)
        print('Preparing T91GamesDataset')

        self.TAGS = {
            'all', 'winning', 'drawish', 'losing',
            'opening', 'middlegame', 'endgame',
            'tactical', 'strategical',
        }
        self.tag = tag.lower()
        if self.tag not in self.TAGS:
            raise ValueError(f'Unknown position tag {tag}; expected one of {self.TAGS}')

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

        self.len = len(self.dataset)

    def _match_tags(self, board, policy, value):
        """Find positions in a game that match the given tag.
        If no positions is found (should be a rare occurence since tags
        are quite general), all positions are marked as valid. This behavior
        can be viewed as using replays since tag filters are currently used
        only in experts post-training.
        """
        if self.tag == 'all':
            return True
        if self.tag in ('winning', 'drawish', 'losing'):
            valuemax = int(value.argmax().item())
            if valuemax == 0:
                return self.tag == 'winning'
            elif valuemax == 1:
                return self.tag == 'drawish'
            else:
                return self.tag == 'losing'

        elif self.tag in ('opening', 'middlegame', 'endgame'):
            return self.tag == get_game_phase(board)

        elif self.tag in ('tactical', 'strategical'):
            max_probability_percent = np.max(policy) if len(policy) else 0.0
            is_tactical = max_probability_percent > 60.0
            return is_tactical if self.tag == 'tactical' else not is_tactical

    def __len__(self):
        return self.len * 10

    def __getitem__(self, idx):
        game = self.dataset[idx % self.len]
        moves = game['movelist'].split()
        fen = game['fen'] if game['fen'] is not None else chess.STARTING_FEN
        board = chess.Board(fen, chess960=True)

        position_count = min(
            len(moves),
            len(game['policies']),
            len(game['qvalues']),
            len(game['dvalues']),
        )
        candidate_positions = list(range(self.min_moves, position_count))

        if self.tag == 'all':
            valid_positions = candidate_positions
        else:
            current_board = board.copy()
            valid_positions = []
            for position_idx, move in enumerate(moves[:position_count]):
                if position_idx >= self.min_moves:
                    policy = np.asarray(game['policies'][position_idx])
                    value, _ = self._get_wdl_and_q(game, current_board.turn, position_idx)
                    if self._match_tags(current_board, policy, value):
                        valid_positions.append(position_idx)
                current_board.push_uci(move)
            if len(valid_positions) == 0:
                valid_positions = candidate_positions # replay

        position_idx = int(np.random.choice(valid_positions))

        for move in moves[:position_idx]:
            board.push_uci(move)

        nodes = self._get_nodes(board, game, position_idx)
        wdl, q = self._get_wdl_and_q(game, board.turn, position_idx)
        sample = process_item(
            board,
            nodes,
            self.encoding,
            value=wdl,
            qvalue=q,
        )
        sample[1]['value_mask'] = torch.ones((), dtype=torch.bool)
        return sample


class LichessPuzzlesDataset(PositionDataset):
    """Dataset of lichess puzzles, used mainly for evaluation. Provides puzzles with
    general (coarse) tags, elo ratings and multiple targets in case of positions with
    multiple mates in 1.

    This dataset is licensed under the cc0-1.0 license
    """
    def __init__(self, encoding, split):
        super().__init__(encoding)
        print('Preparing LichessPuzzlesDataset')
        if split not in ('train', 'valid'):
            raise ValueError(f'LichessPuzzlesDataset: split should be train or valid, received {split}')
        self.split = split

        self.TAGS = {
            'defensiveMove', 'zugzwang', 'sacrifice', 'mate', 'quietMove', 'fork',
            'skewer', 'pin', 'intermezzo', 'clearance', 'attraction', 'xRayAttack',
            'advancedPawn', 'opening', 'middlegame', 'endgame',
        }

        self.dataset = load_dataset('Lichess/chess-puzzles', split='train')
        if split == 'valid':
            selected = range(100_000)
        elif split == 'train':
            selected = range(1_000_000, 6_000_000)
        else:
            raise ValueError(f'LichessPuzzleDataset: split should be train or valid, received {split}')

        selected_themes = self.TAGS | {''}

        self.dataset = self.dataset.select(selected).shuffle().map(
            lambda x: {
                'fen': x['FEN'],
                'moves': x['Moves'].split(' '),
                'rating': x['Rating'],
                'themes': [
                    tag for tag in (
                        x['Themes'].split()
                        if isinstance(x['Themes'], str)
                        else x['Themes']
                    )
                    if tag in selected_themes
                ],
                'puzzle_len': len(x['Moves'].split(' ')) // 2
            }, num_proc=16
        )

        self.puzzles_lengths = np.cumsum(self.dataset['puzzle_len'])
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

        if self.split == 'train':
            return_value = torch.full((3,), 1 / 3, dtype=torch.float)
            sample = process_item(
                board,
                [VariationNode(move=uci_move, probability=1.0)],
                self.encoding,
                self.temperature,
                value=return_value,
            )
            sample[1]['value_mask'] = torch.zeros((), dtype=torch.bool)
            return sample

        board_, extra_ = encode_board(board, self.encoding)
        turn = board.turn
        move_id = get_move_id(chess.Move.from_uci(uci_move), turn)

        mating_move_ids = []
        for move in list(board.legal_moves):
            board.push(move)
            if board.is_checkmate():
                mating_move_ids.append(get_move_id(move, turn))
            board.pop()

        acceptable_move_ids = sorted(set([move_id] + mating_move_ids))
        confidence_move_ids = mating_move_ids or [move_id]
        tags = torch.tensor(
            [tag in puzzle['themes'] for tag in self.TAGS],
            dtype=torch.bool,
        )

        move_id_ = torch.tensor(move_id, dtype=torch.long)
        acceptable_move_ids_ = torch.tensor(acceptable_move_ids, dtype=torch.long)
        confidence_move_ids_ = torch.tensor(confidence_move_ids, dtype=torch.long)
        puzzle_idx_ = torch.tensor(puzzle_idx, dtype=torch.long)
        rating_ = torch.tensor(float(puzzle['rating']), dtype=torch.float32)

        return (
            board_, move_id_, puzzle_idx_, rating_, extra_,
            acceptable_move_ids_, confidence_move_ids_, tags,
        )


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
