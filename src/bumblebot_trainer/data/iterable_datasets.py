from torch.utils.data import IterableDataset
from datasets import load_dataset, VerificationMode
from huggingface_hub import list_repo_tree
import chess

from .utils import process_item, VariationNode, san_to_uci


def get_lichess_files_for_year(year: int) -> list[str]:
    files = list_repo_tree(
        'Lichess/standard-chess-games',
        repo_type='dataset',
        path_in_repo=f'data/year={year}',
        recursive=True,
    )
    return sorted(
        f.path for f in files if f.path.endswith('.parquet')
    )


class IterablePositionDataset(IterableDataset):
    def __init__(self, encoding: str):
        self.encoding = encoding
        self.temperature = 0.1

    def __iter__(self):
        raise NotImplementedError()


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

        data_files = get_lichess_files_for_year(2025)

        self.dataset = load_dataset(
            'Lichess/standard-chess-games',
            split='train',
            data_files=data_files,
            streaming=True
        )

    def __iter__(self):
        for item in self.dataset:
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

