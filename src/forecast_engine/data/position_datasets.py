from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset
import chess
import chess.syzygy
from datasets import load_dataset, VerificationMode
from datasets_sql import query as sql_query
import pandas as pd

from ..utils import encode_board, eval_to_whitewinpercent, get_move_id, ChessConstants, ForecastVocabulary

from typing import Optional


@dataclass
class VariationNode:
    move: str
    cp: Optional[int]
    mate: Optional[int]
    expected_result: Optional[float]


class PositionDataset(Dataset):
    def __init__(self, forecast_depth: int):
        super().__init__()

        self.ignore_index = -100
        self.forecast_depth = forecast_depth

        self.dataset = ...

        self.temperature = 0.1

        self.taken_cls = 64
        self.promotion_cls_map = {
            chess.QUEEN: 65,
            chess.ROOK: 66,
            chess.BISHOP: 67,
            chess.KNIGHT: 68
        }

    def __len__(self):
        return len(self.dataset)

    def _process_item(
            self,
            board: chess.Board,
            nodes: list[VariationNode],
            forecast: Optional[torch.Tensor] = None,
            forecast_mask: Optional[torch.Tensor] = None):
        tokens, hm, ep_square = encode_board(board)

        indices = torch.zeros((len(nodes),), dtype=torch.long)
        evals = torch.zeros((len(nodes),), dtype=torch.float)

        for i, node in enumerate(nodes):
            if node.expected_result is not None:
                expected_result = node.expected_result
            else:
                winpercent = eval_to_whitewinpercent(node.cp, node.mate)
                if board.turn == chess.BLACK and winpercent != -100:
                    winpercent = 100.0 - winpercent
                expected_result = winpercent / 100.0

            indices[i] = get_move_id(chess.Move.from_uci(node.move), board.turn)
            evals[i] = expected_result

        policy_target = torch.full((ChessConstants.NUM_POLICY_CLASSES,), float('-inf'), dtype=torch.float)
        policy_target[indices] = evals
        policy_target = policy_target / self.temperature
        policy_target = torch.softmax(policy_target, dim=-1)

        value_target = torch.max(evals)

        if forecast is None:
            assert forecast_mask is None, 'Malformed input: forecast'
            forecast = torch.zeros(self.forecast_depth * 64, dtype=torch.long)
            forecast_mask = torch.zeros(self.forecast_depth * 64, dtype=torch.bool)

        target_dict = {
            'policy': policy_target,
            'value': value_target,
            'forecast': forecast,
            'forecast_mask': forecast_mask
        }

        return (
            tokens,
            hm,
            ep_square,
            target_dict,
        )

    def _get_forecast(self, board: chess.Board, moves: list[str]):
        black_to_move = (board.turn == chess.BLACK)
        if black_to_move:
            board.apply_mirror()

        forecast_data = [[] for _ in range(64)]
        horizon_data = [[] for _ in range(64)]
        location_dict = {square: square for square in chess.SQUARES if board.piece_at(square) is not None}

        def get_horizon_cls(move_idx: int) -> int:
            if move_idx < 10:
                return 0 # short-term
            elif move_idx < 20:
                return 1 # mid-term
            else:
                return 2 # long-term

        for i, uci_move in enumerate(moves):
            move = chess.Move.from_uci(uci_move)
            if black_to_move:
                move.from_square = chess.square_mirror(move.from_square)
                move.to_square = chess.square_mirror(move.to_square)

            if board.is_castling(move):
                if board.is_kingside_castling(move): # g-file: kingside
                    rook_offset = 7 # rook is to the right of the king (white's perspective)
                    rook_end_offset = -1 # rook ends up on the f-file
                    king_offset = 6 # king ends up on the g-file
                elif board.is_queenside_castling(move): # c-file: queenside
                    rook_offset = 0 # rook is to the left of the king (white's perspective)
                    rook_end_offset = 1 # rook ends up on the d-file
                    king_offset = 2 # king ends up on the c-file
                else:
                    raise ValueError(f"Invalid castling move: {uci_move}")

                target_piece = board.piece_at(move.to_square)
                if target_piece is not None and target_piece.piece_type == chess.ROOK: # chess960 castling
                    rook_square = move.to_square
                else: # standard chess castling
                    rook_square = 8 * (move.from_square // 8) + rook_offset

                king_start_square = location_dict.pop(move.from_square)
                king_end_square = 8 * (move.from_square // 8) + king_offset

                forecast_data[king_start_square].append(king_end_square)
                horizon_data[king_start_square].append(get_horizon_cls(i))

                location_dict[king_end_square] = king_start_square

                rook_start_square = location_dict.pop(rook_square)
                rook_end_square = king_end_square + rook_end_offset

                forecast_data[rook_start_square].append(rook_end_square)
                horizon_data[rook_start_square].append(get_horizon_cls(i))

                location_dict[rook_end_square] = rook_start_square

                board.push(move)
                continue

            if board.is_en_passant(move):

                capture_square = move.to_square + (-8 if board.turn == chess.WHITE else 8)

                start_square = location_dict.pop(capture_square)

                forecast_data[start_square].append(self.taken_cls)
                horizon_data[start_square].append(get_horizon_cls(i))

            if move.to_square in location_dict:
                start_square = location_dict.pop(move.to_square)

                forecast_data[start_square].append(self.taken_cls)
                horizon_data[start_square].append(get_horizon_cls(i))

            if move.from_square in location_dict:
                start_square = location_dict.pop(move.from_square)

                forecast_data[start_square].append(move.to_square)
                horizon_data[start_square].append(get_horizon_cls(i))

                location_dict[move.to_square] = start_square

                if move.promotion is not None:
                    promoted_piece_type = move.promotion
                    forecast_data[start_square].append(self.promotion_cls_map[promoted_piece_type])
                    horizon_data[start_square].append(get_horizon_cls(i))

            board.push(move)

        for i in range(64):
            forecast = forecast_data[i]
            horizon = horizon_data[i]
            if len(forecast) > self.forecast_depth:
                forecast = forecast[:self.forecast_depth]
                horizon = horizon[:self.forecast_depth]
            forecast += [self.ignore_index] * (self.forecast_depth - len(forecast))
            horizon += [self.ignore_index] * (self.forecast_depth - len(horizon))

            forecast_data[i] = forecast
            horizon_data[i] = horizon

        forecast_data = torch.as_tensor(forecast_data, dtype=torch.long)
        forecast_data = forecast_data.permute(1, 0) # forecast_depth x 64

        forecast_mask = (forecast_data != self.ignore_index)
        forecast_data = torch.where(forecast_data == self.ignore_index, self.taken_cls, forecast_data)


        horizon_data = torch.as_tensor(horizon_data, dtype=torch.long)
        horizon_data = horizon_data.permute(1, 0) # forecast_depth x 64
        horizon_data = torch.where(horizon_data == self.ignore_index, 0, horizon_data)

        forecast_data = forecast_data + horizon_data * ForecastVocabulary.HORIZON_OFFSET
        return forecast_data, forecast_mask

    def __getitem__(self, index):
        return NotImplementedError()


class CombinedPositionDataset(Dataset):
    def __init__(self, datasets: list[PositionDataset]):
        super().__init__()

        self.datasets = datasets
        self.cumulative_sizes = np.cumsum([len(ds) for ds in datasets])

    def __len__(self):
        return int(self.cumulative_sizes[-1])

    def __getitem__(self, index):
        dataset_idx = np.searchsorted(self.cumulative_sizes, index, side='right')

        if dataset_idx == 0:
            sample_idx = index
        else:
            sample_idx = index - self.cumulative_sizes[dataset_idx - 1]

        return self.datasets[dataset_idx][sample_idx]


class PuzzleDataset(PositionDataset):
    def __init__(self, forecast_depth, split='train'):
        super().__init__(forecast_depth)
        print('Preparing PuzzleDataset')

        self.dataset = load_dataset('Lichess/chess-puzzles', split='train')
        # dataset of 5.7m puzzles, ~9.8m positions

        if split == 'train':
            indices = range(5_000_000)
        elif split == 'val':
            indices = range(5_000_000, 5_250_000)
        elif split == 'test':
            indices = range(5_250_000, 5_500_000)
        else: raise ValueError(f'Unknown split: {split}')

        self.dataset = self.dataset.select(
            indices
        )
        self.dataset = self.dataset.map(
            lambda x: {'FEN': x['FEN'], 'Moves': x['Moves'],
                       'MoveCount': len(x['Moves'].split(' ')) // 2},
            num_proc=16
        )

        self.puzzles_moves = np.cumsum(
            np.asarray(self.dataset['MoveCount'])
        )

    def __len__(self):
        return int(self.puzzles_moves[-1])

    def __getitem__(self, index):
        puzzle_idx = np.searchsorted(self.puzzles_moves, index, side='right')

        if puzzle_idx == 0:
            move_idx = index
        else:
            move_idx = index - self.puzzles_moves[puzzle_idx - 1]

        sample = self.dataset[int(puzzle_idx)]

        board = chess.Board(sample['FEN'])

        # this dataset contains (position, line) pairs
        # we only keep the positions that are intended for the player
        # to have proper puzzle solving training data
        moves = sample['Moves'].split(' ')
        for i in range(2 * move_idx + 1):
            board.push_uci(moves[i])

        uci_move = moves[2 * move_idx + 1]

        nodes = [
            VariationNode(uci_move, cp=None, mate=None, expected_result=None)
        ]

        return self._process_item(board, nodes)


class LichessEvalDataset(PositionDataset):
    def __init__(self, forecast_depth, split='train'):
        super().__init__(forecast_depth)
        print('Preparing LichessEvalDataset')

        data_files = [
            'data/train-00000-of-00017.parquet',
            'data/train-00001-of-00017.parquet',
            # 'data/train-00002-of-00017.parquet',
            # 'data/train-00003-of-00017.parquet',
            # 'data/train-00004-of-00017.parquet',
            # 'data/train-00005-of-00017.parquet',
            # 'data/train-00006-of-00017.parquet',
            # 'data/train-00007-of-00017.parquet',
            # 'data/train-00008-of-00017.parquet',
            # 'data/train-00009-of-00017.parquet',
            # 'data/train-00010-of-00017.parquet',
            # 'data/train-00011-of-00017.parquet',
            # 'data/train-00012-of-00017.parquet',
            # 'data/train-00013-of-00017.parquet',
            # 'data/train-00014-of-00017.parquet',
            # 'data/train-00015-of-00017.parquet',
            # 'data/train-00016-of-00017.parquet',
        ]

        dataset = load_dataset(
            'Lichess/chess-position-evaluations',
            data_files=data_files,
            split='train',
            verification_mode=VerificationMode.NO_CHECKS
        )
        dataset = sql_query(
        """
            SELECT fen, LIST(line) as lines, LIST(depth) as depths, LIST(cp) as cps, LIST(mate) as mates
            FROM dataset
            GROUP BY fen
            ORDER BY fen
        """, load_from_cache_file=False
        )
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        board = chess.Board(sample['fen'])
        lines = sample['lines']
        cps = sample['cps']
        mates = sample['mates']
        depths = sample['depths']

        max_depth = max(depths)

        nodes = []

        for line, cp_, mate_, depth in zip(lines, cps, mates, depths):
            if depth == max_depth:
                nodes.append(VariationNode(line.split(' ')[0], cp=cp_, mate=mate_, expected_result=None))

        return self._process_item(board, nodes)


class TablebaseDataset(PositionDataset):
    def __init__(self, forecast_depth, tablebase_path, fens_path):
        super().__init__(forecast_depth)
        print('Preparing TablebaseDataset')

        self.tablebase_path = tablebase_path
        self.fens = pd.read_parquet(fens_path)
        self.tb = chess.syzygy.open_tablebase(tablebase_path)

    def __len__(self):
        return len(self.fens)

    def _get_expected_result(self, wdl):
        if wdl == -2:
            return 0.0
        elif wdl == -1:
            return 0.25
        elif wdl == 0:
            return 0.5
        elif wdl == 1:
            return 0.75
        elif wdl == 2:
            return 1.0
        else:
            return -1.0

    def __getitem__(self, index):
        fen = self.fens.iloc[index]['fen']
        board = chess.Board(fen)

        lines = []
        for move in board.legal_moves:
            board.push(move)

            # negate the expected_result since perspective changed
            expected_result = -self._get_expected_result(self.tb.probe_wdl(board))
            board.pop()

            node = VariationNode(
                move=move.uci(), cp=None, mate=None, expected_result=expected_result
            )
            lines.append(node)

        return self._process_item(board, lines)


class SinglePositionDataset(PositionDataset):
    def __init__(self, start_fen, forecast_depth):
        super().__init__(forecast_depth)

        self.start_fen = start_fen
        self.forecast_depth = forecast_depth

        self.board = chess.Board(start_fen)

    def __len__(self):
        return 1
    
    def __getitem__(self, index):
        return self._process_item(self.board, [])