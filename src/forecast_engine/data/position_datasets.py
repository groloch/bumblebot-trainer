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
    def __init__(self):
        super().__init__()

        self.ignore_index = -100

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
            nodes: list[VariationNode]):
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

        target_dict = {
            'policy': policy_target,
            'value': value_target,
        }

        return (
            tokens,
            hm,
            ep_square,
            target_dict,
        )

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


class SinglePositionDataset(PositionDataset):
    def __init__(self, start_fen):
        super().__init__()

        self.start_fen = start_fen

        self.board = chess.Board(start_fen)

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self._process_item(self.board, [])