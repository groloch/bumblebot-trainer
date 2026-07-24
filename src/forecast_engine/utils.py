import numpy as np
import torch
import chess


class ChessVocabulary:
    PIECE_TOKENS = {
        None: 0,
        chess.Piece(chess.PAWN, chess.WHITE): 1, chess.Piece(chess.KNIGHT, chess.WHITE): 2,
        chess.Piece(chess.BISHOP, chess.WHITE): 3, chess.Piece(chess.ROOK, chess.WHITE): 4,
        chess.Piece(chess.QUEEN, chess.WHITE): 5, chess.Piece(chess.KING, chess.WHITE): 6,
        chess.Piece(chess.PAWN, chess.BLACK): 7, chess.Piece(chess.KNIGHT, chess.BLACK): 8,
        chess.Piece(chess.BISHOP, chess.BLACK): 9, chess.Piece(chess.ROOK, chess.BLACK): 10,
        chess.Piece(chess.QUEEN, chess.BLACK): 11, chess.Piece(chess.KING, chess.BLACK): 12,
    }
    CASTLING_BASE = 13
    HALFMOVE_TOKEN = 21
    CLS_TOKEN_ID = 22
    TOTAL_TOKENS = 23


class ChessConstants:
    NUM_POLICY_CLASSES = 4288 # 64x64 + 8x8x3 for promotions
    CONTEXT_LENGTH = 70 # 64 squares + 1 CLS + 5 registers
    N_FORECAST_CLASSES = 69 # 64 squares + 1 taken logit + 4 promotion logits
    N_HORIZON_CLASSES = 3 # 0: close term, 1: medium term, 2: long term
    MAX_NUMBER_OF_MOVES = 256


class ForecastVocabulary:
    SQUARE_TOKENS = {i: i for i in range(64)}
    TAKEN_TOKEN = 64
    PROMOTION_TOKENS = {chess.ROOK: 65, chess.BISHOP: 66, chess.KNIGHT: 67, chess.QUEEN: 68}
    HORIZON_OFFSET = 69

    PER_HORIZON_CTX_LENGTH = 64

    @classmethod
    def NUM_FORECAST_CLASSES(cls, forecast_depth):
        return 1 + cls.HORIZON_OFFSET * forecast_depth
    
    @classmethod
    def MASK_TOKEN_ID(cls, forecast_depth):
        return cls.HORIZON_OFFSET * forecast_depth


def encode_board(board: chess.Board):
    if board.turn == chess.BLACK:
        board = board.mirror()
    tokens = [ChessVocabulary.PIECE_TOKENS[board.piece_at(s)] for s in range(64)]
    tokens += [
        ChessVocabulary.CLS_TOKEN_ID,
        ChessVocabulary.CASTLING_BASE + (0 if board.has_kingside_castling_rights(chess.WHITE) else 1),
        ChessVocabulary.CASTLING_BASE + (2 if board.has_queenside_castling_rights(chess.WHITE) else 3),
        ChessVocabulary.CASTLING_BASE + (4 if board.has_kingside_castling_rights(chess.BLACK) else 5),
        ChessVocabulary.CASTLING_BASE + (6 if board.has_queenside_castling_rights(chess.BLACK) else 7),
        ChessVocabulary.HALFMOVE_TOKEN
    ]

    ep_square = board.ep_square
    if ep_square is None:
        ep_square = -1

    return (
        torch.tensor(tokens, dtype=torch.long),
        torch.tensor(min(board.halfmove_clock, 50)/50.0, dtype=torch.float),
        torch.tensor(ep_square, dtype=torch.long)
    )

def eval_to_whitewinpercent(cp: int, mate: int, magic=0.00368208):
    """Converts position evaluation to win% (white's perspective) according to:
    https://lichess.org/page/accuracy
    """
    if mate is not None:
        if mate > 0:
            return 100
        else:
            return 0
    if cp is None:
        return -100
    winpercent = np.rint(100 / (1 + np.exp(-magic * cp)))
    return max(min(winpercent, 100), 0)

def whitewinpercent_to_cp(winpercent: float, magic=0.00368208):
    """Converts position evaluation from win% (white's perspective) to centipawns according to:
    https://lichess.org/page/accuracy
    """
    if winpercent <= 0:
        return 1000
    elif winpercent >= 1:
        return -1000
    cp = -np.log(1 / winpercent - 1) / magic
    cp = max(min(cp, 1000), -1000)
    return int(cp)

def get_move_id(move: chess.Move, turn:chess.Color):
    from_square = move.from_square
    to_square = move.to_square
 
    if turn == chess.BLACK: # white's perspective
        from_square = chess.square_mirror(from_square)
        to_square = chess.square_mirror(to_square)

    piece_to_idx = {
        chess.ROOK: 0,
        chess.BISHOP: 1,
        chess.KNIGHT: 2
    }

    if move.promotion is None or move.promotion == chess.QUEEN:
        return from_square * 64 + to_square
    else:
        assert move.promotion in (chess.ROOK, chess.BISHOP, chess.KNIGHT), "Invalid promotion piece"
        return 4096 + (from_square - 48) * 24 + (to_square - 56) * 3 + piece_to_idx[move.promotion]

def get_move_from_id(index: int, turn: chess.Color) -> chess.Move:
    idx_to_piece = {0: chess.ROOK, 1: chess.BISHOP, 2: chess.KNIGHT}

    if index < 4096:
        from_square = index // 64
        to_square = index % 64
        # rank 7 -> rank 8 in white's perspective is a pawn promotion (queen by convention)
        promotion = chess.QUEEN if (48 <= from_square <= 55 and 56 <= to_square <= 63) else None
    else:
        idx = index - 4096
        from_square = idx // 24 + 48
        remaining = idx % 24
        to_square = remaining // 3 + 56
        promotion = idx_to_piece[remaining % 3]

    if turn == chess.BLACK:  # undo white's perspective flip
        from_square = chess.square_mirror(from_square)
        to_square = chess.square_mirror(to_square)

    return chess.Move(from_square, to_square, promotion=promotion)

def model_parameters(model: torch.nn.Module):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total_params, trainable_params
