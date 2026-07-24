import regex as re

import chess


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
