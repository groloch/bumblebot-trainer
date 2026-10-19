import json

import numpy as np
import torch
from torch.utils.data import Dataset
import chess
from datasets import load_dataset, VerificationMode
from datasets_sql import query as sql_query

from .utils import process_item, get_game_phase, VariationNode
from ..utils import eval_to_whitewinpercent


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


class LichessEvalDataset(PositionDataset):
    def __init__(self, encoding, split='train'):
        super().__init__(encoding)
        print('Preparing LichessEvalDataset')

        data_files = [
            'data/data_0000.parquet',
            'data/data_0001.parquet',
            # 'data/data_0002.parquet',
            # 'data/data_0003.parquet',
            # 'data/data_0004.parquet',
            # 'data/data_0005.parquet',
            # 'data/data_0006.parquet',
            # 'data/data_0007.parquet',
            # 'data/data_0008.parquet',
            # 'data/data_0009.parquet',
            # 'data/data_0010.parquet',
            # 'data/data_0011.parquet',
            # 'data/data_0012.parquet',
            # 'data/data_0013.parquet',
            # 'data/data_0014.parquet',
            # 'data/data_0015.parquet',
            # 'data/data_0016.parquet',
        ]

        if split == 'train':
            selected = range(1_000_000, 30_000_000)
        elif split == 'eval':
            selected = range(500_000)

        dataset = load_dataset(
            'Lichess/chess-position-evaluations',
            data_files=data_files,
            split='train',
            verification_mode=VerificationMode.NO_CHECKS,
        )
        dataset = dataset.shuffle().select(selected).flatten_indices(num_proc=16)

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
        max_depth = max(sample['depths'])

        scored_nodes = []
        for line, cp, mate, depth in zip(
            sample['lines'], sample['cps'], sample['mates'], sample['depths']
        ):
            if depth != max_depth or not line or (cp is None and mate is None):
                continue

            node = VariationNode(move=line.split()[0], cp=cp, mate=mate)
            winpercent = eval_to_whitewinpercent(cp, mate)
            if board.turn == chess.BLACK and winpercent != -100:
                winpercent = 100.0 - winpercent
            expected_result = winpercent / 100.0
            scored_nodes.append((node, expected_result))

        if not scored_nodes:
            raise ValueError(f'LichessEvalDataset position {index} has no scored PV at max depth')

        nodes = [node for node, _ in scored_nodes]
        best_node, best_expected_result = max(scored_nodes, key=lambda item: item[1])
        if best_node.mate is not None:
            stockfish_cp = 1000.0 if best_expected_result >= 0.5 else -1000.0
        else:
            stockfish_cp = float(best_node.cp)
            if board.turn == chess.BLACK:
                stockfish_cp = -stockfish_cp
        stockfish_cp = max(min(stockfish_cp, 1000.0), -1000.0)

        tokens, targets, extra = process_item(
            board, nodes, self.encoding, self.temperature
        )
        targets['stockfish_cp'] = torch.tensor(stockfish_cp, dtype=torch.float32)
        targets['game_phase'] = torch.tensor(
            ('opening', 'middlegame', 'endgame').index(get_game_phase(board)),
            dtype=torch.long,
        )
        return tokens, targets, extra


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
