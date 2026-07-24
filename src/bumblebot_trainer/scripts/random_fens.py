"""
Generates random chess positions with a fixed number of pieces for tablebase probing.
"""

import random
from multiprocessing import Pool

import chess
from tqdm import tqdm
import pandas as pd


def generate_random_fens(n, k):
    fens = set()
    pbar = tqdm(total=n, desc=f'Generating {k}-piece FENs')
    while len(fens) < n:
        board = chess.Board.empty()
        occupied_squares = set()

        wk_sq = 0
        bk_sq = 0
        while wk_sq == bk_sq:
            wk_sq = random.randint(0, 63)
            bk_sq = random.randint(0, 63)

        board.set_piece_at(wk_sq, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(bk_sq, chess.Piece(chess.KING, chess.BLACK))

        occupied_squares.add(wk_sq)
        occupied_squares.add(bk_sq)

        for _ in range(k-2):
            piece_type = random.choice([chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN])
            piece_color = random.choice([chess.WHITE, chess.BLACK])
            piece_sq = -1
            while piece_sq == -1 or piece_sq in occupied_squares:
                piece_sq = random.randint(0, 63)
            board.set_piece_at(piece_sq, chess.Piece(piece_type, piece_color))
            occupied_squares.add(piece_sq)

        if board.is_valid() and not board.is_game_over(claim_draw=True):
            fens.add(board.fen())
            pbar.update(1)

    pbar.close()
    return fens

def generate_random_3piece_fens(n):
    print('Generating 3-piece FENs...')
    fens = list(generate_random_fens(n, 3))
    print(f'Generated {len(fens)} unique 3-piece FENs')
    return fens

def generate_random_4piece_fens(n):
    print('Generating 4-piece FENs...')
    with Pool(10) as pool:
        results = pool.starmap(generate_random_fens, [(n // 10, 4)] * 10)
    fens = list({fen for sublist in results for fen in sublist})
    print(f'Generated {len(fens)} unique 4-piece FENs')
    return fens

def generate_random_5piece_fens(n):
    print('Generating 5-piece FENs...')
    with Pool(10) as pool:
        results = pool.starmap(generate_random_fens, [(n // 10, 5)] * 10)
    fens = list({fen for sublist in results for fen in sublist})
    print(f'Generated {len(fens)} unique 5-piece FENs')
    return fens

def generate_data(n3, n4, n5):
    fens_3piece = generate_random_3piece_fens(n3)
    fens_4piece = generate_random_4piece_fens(n4)
    fens_5piece = generate_random_5piece_fens(n5)
    df = pd.DataFrame(
        [(fen, 3) for fen in fens_3piece] +
        [(fen, 4) for fen in fens_4piece] +
        [(fen, 5) for fen in fens_5piece],
        columns=['fen', 'n']
    )
    df.to_parquet('tb/fens.parquet', index=False)


if __name__ == '__main__':
    generate_data(n3=100_000, n4=1_000_000, n5=10_000_000)
