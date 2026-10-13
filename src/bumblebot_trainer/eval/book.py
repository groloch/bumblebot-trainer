"""Load opening positions from supported book formats."""

import csv
from pathlib import Path

import chess.pgn


def load_openings(book_path: Path) -> list[str]:
    if not book_path.is_file():
        raise FileNotFoundError(f'Opening book not found: {book_path}')

    openings = []
    if book_path.suffix.lower() == '.pgn':
        with book_path.open(encoding='utf-8-sig') as book_file:
            game_number = 0
            while (game := chess.pgn.read_game(book_file)) is not None:
                game_number += 1
                if game.errors:
                    raise ValueError(
                        f'Could not parse PGN game {game_number} in {book_path}: '
                        f'{game.errors[0]}'
                    )
                board = game.end().board()
                if next(iter(board.legal_moves), None) is not None:
                    openings.append(board.fen())
    else:
        with book_path.open(newline='') as book_file:
            for row in csv.reader(book_file, delimiter=';'):
                if row and row[0].strip():
                    openings.append(row[0].strip())

    if not openings:
        raise ValueError(f'No starting positions found in opening book {book_path}')
    return openings
