import ctypes
import gc

from torch.utils.data import IterableDataset
from datasets import load_dataset
import chess
import numpy as np
import pyarrow as pa
import pyarrow.dataset as arrow_dataset

from .utils import (
    process_item,
    VariationNode,
    san_to_uci,
    get_lichess_files_for_year,
    T91_REPO_ID,
    WdlDataset
)


# Arrow's pool and libc can both retain freed Parquet decode/read buffers.
# malloc_trim is available on this training host (glibc); other platforms still
# benefit from releasing Arrow's pool and collecting unreachable file objects.
try:
    _malloc_trim = getattr(ctypes.CDLL(None), 'malloc_trim', None)
except OSError:
    _malloc_trim = None
if _malloc_trim is not None:
    _malloc_trim.argtypes = (ctypes.c_size_t,)
    _malloc_trim.restype = ctypes.c_int


class IterablePositionDataset(IterableDataset):
    def __init__(self, encoding: str):
        self.encoding = encoding
        self.temperature = 0.1

    def __iter__(self):
        raise NotImplementedError()

    def _iter_items(self):
        for index, item in enumerate(self.dataset):
            # Parquet decoding can leave large freed allocations in Arrow's pool.
            # Return them to the OS regularly: each DataLoader worker has its own
            # pool, and eight retained high-water marks can exhaust system RAM.
            if index % 256 == 0:
                gc.collect()
                pa.default_memory_pool().release_unused()
                if _malloc_trim is not None:
                    _malloc_trim(0)
            yield item


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

        # Cap the total before HF resolves files or reads Parquet metadata.
        data_files = []
        for year in (2025, 2024, 2023):
            remaining = 100 - len(data_files)
            if remaining <= 0:
                break
            data_files.extend(get_lichess_files_for_year(year, limit=remaining))

        self.dataset = load_dataset(
            'Lichess/standard-chess-games',
            split='train',
            data_files=data_files,
            columns=['WhiteElo', 'BlackElo', 'movetext'],
            streaming=True,
            batch_size=32,
            # Avoid retaining compressed data from previously scanned row groups.
            fragment_scan_options=arrow_dataset.ParquetFragmentScanOptions(pre_buffer=False),
        )
        print(f'Lichess streaming: {len(data_files)} files, {self.dataset.num_shards} shards')

    def __iter__(self):
        for item in self._iter_items():
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
    """groloch/lc0_selfplay_games is a dataset of games played Leela Chess Zero
     against itself during its extended RL runs (test91).

    This dataset is licensed under the odbl licence
    """
    def __init__(
            self,
            min_moves: int,
            encoding: str,
            shuffle_buffer_size: int = 128,
            seed: int = 0,
            num_workers: int = 0
        ):
        super().__init__(encoding)
        WdlDataset.__init__(self)

        self.min_moves = min_moves

        self.dataset = load_dataset(
            T91_REPO_ID,
            columns=['fen', 'movelist', 'policies', 'qvalues', 'dvalues'],
            split='train',
            streaming=True,
            batch_size=32,
            # Avoid retaining compressed data from previously scanned row groups.
            fragment_scan_options=arrow_dataset.ParquetFragmentScanOptions(pre_buffer=False),
        )

        # Only inspect row-group metadata when file shards cannot fill the workers.
        # Do this before shuffle so HF can assign disjoint shards to each worker.
        if self.dataset.num_shards < num_workers:
            self.dataset = self.dataset.reshard()
        self.dataset = self.dataset.shuffle(
            seed=seed, buffer_size=shuffle_buffer_size, max_buffer_input_shards=1,
        )
        print(f'T91 streaming: {self.dataset.num_shards} shards, '
              f'{num_workers} workers, shuffle buffer {shuffle_buffer_size} games/worker')

    def __iter__(self):
        while True:
            saw_item = False
            for item in self._iter_items():
                saw_item = True
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
                value, qvalue = self._get_wdl_and_q(item, board.turn, move_idx)
                yield process_item(
                    board,
                    nodes,
                    self.encoding,
                    self.temperature,
                    value=value,
                    qvalue=qvalue,
                )

            # HF assigns no shards to excess workers; let those workers stop.
            if not saw_item:
                return
