import sys
import regex as re

import torch
import chess

from typing import Literal


def san_to_uci(movetext: str, start_fen: str = chess.STARTING_FEN, chess960: bool = False) -> list[str]:
    """Convert a san movetext string to a list of uci moves."""
    movetext = re.sub(r'\{[^}]*\}', '', movetext)
    board = chess.Board(start_fen, chess960=chess960)
    uci_moves = []
    for token in movetext.split():
        if token.endswith('.') or token in ('1-0', '0-1', '1/2-1/2', '*'):
            continue
        token = token.rstrip('?!')
        move = board.parse_san(token)
        uci_moves.append(move.uci())
        board.push(move)
    return uci_moves

def encode_board(
        board: chess.Board,
        encoding_type: Literal['simplified', 'lc0'] = 'simplified'
        ) -> torch.Tensor:
    """Encodes a board state into tokens for the specified encoding type
    """
    black_turn = False
    if board.turn == chess.BLACK:
        black_turn = True
        board = board.mirror()

    if encoding_type == 'lc0':
        tokens = torch.zeros((64, 112), dtype=torch.float32)

        b = board.copy()

        for k in range(8):
            offset = 13 * k

            if b.is_repetition(2):
                tokens[:, offset + 12] = 1.0

            for square in chess.SQUARES:
                piece = b.piece_at(square)
                if piece is not None:
                    piece_type = piece.piece_type - 1
                    color_offset = 0 if piece.color == chess.WHITE else 6
                    tokens[square, offset + piece_type + color_offset] = 1.0

            if len(b.move_stack) > 0:
                b.pop()
            else:
                break

        if board.has_queenside_castling_rights(chess.WHITE):
            tokens[:, 104] = 1.0
        if board.has_kingside_castling_rights(chess.WHITE):
            tokens[:, 105] = 1.0
        if board.has_queenside_castling_rights(chess.BLACK):
            tokens[:, 106] = 1.0
        if board.has_kingside_castling_rights(chess.BLACK):
            tokens[:, 107] = 1.0

        if black_turn:
            tokens[:, 108] = 1.0

        tokens[:, 109] = board.halfmove_clock / 100.0
        tokens[:, 111] = 1.0

        return tokens

    if encoding_type == 'simplified':
        tokens = torch.zeros((64, 18), dtype=torch.float32)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                piece_type = piece.piece_type - 1
                color_offset = 0 if piece.color == chess.WHITE else 6
                tokens[square, piece_type + color_offset] = 1.0
        if board.has_kingside_castling_rights(chess.WHITE):
            tokens[:, 12] = 1.0
        if board.has_queenside_castling_rights(chess.WHITE):
            tokens[:, 13] = 1.0
        if board.has_kingside_castling_rights(chess.BLACK):
            tokens[:, 14] = 1.0
        if board.has_queenside_castling_rights(chess.BLACK):
            tokens[:, 15] = 1.0
        tokens[:, 16] = board.halfmove_clock / 100.0
        if board.ep_square is not None:
            tokens[board.ep_square, 17] = 1.0
        return tokens

    raise ValueError(f"Unknown encoding type: {encoding_type}")
    return None
