from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset
import chess

from .utils import encode_board, process_item, VariationNode
from ..utils import eval_to_whitewinpercent, get_move_id, ChessConstants

from typing import Optional


class PositionDataset(Dataset):
    """Abstract class for datasets of positions.
    """
    def __init__(self, encoding: str):
        super().__init__()

        self.ignore_index = -100

        self.dataset = ...

        self.temperature = 0.1

        self.encoding = encoding

        self.taken_cls = 64
        self.promotion_cls_map = {
            chess.QUEEN: 65,
            chess.ROOK: 66,
            chess.BISHOP: 67,
            chess.KNIGHT: 68
        }

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        raise NotImplementedError()


class CombinedPositionDataset(Dataset):
    """Utility class to aggregate several position datasets from different sources.
    """
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
    """For testing purposes, a dataset made of a single position
    """
    def __init__(self, start_fen, encoding):
        super().__init__(encoding)

        self.start_fen = start_fen

        self.board = chess.Board(start_fen)

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return process_item(self.board, [], self.encoding, self.temperature)
