import json

import numpy as np
from torch.utils.data import Dataset
import chess
from datasets import load_dataset

from .utils import process_item, VariationNode


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


class Lc0PositionDataset(PositionDataset):
    """Dataset of positions from lc0 T91 training chunks.
    """

    def __init__(self, directory: str, encoding: str):
        super().__init__(encoding)

        data_files = [
            f'training-run2-test91-20251118-{x}{y}17.parquet'
            for x in range(2)
            for y in range(10)
        ]

        self.dataset = load_dataset(
            'Maxlegrec/test91-data',
            split='train',
            data_files=data_files
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pass