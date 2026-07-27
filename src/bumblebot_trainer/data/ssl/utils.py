import torch
import chess

from ..utils import encode_board, get_move_id
from ...utils import ChessConstants


class SSLConstants:
    NUM_TOKENS_PER_MOVE = 6


class SSLCollator:
    """Custom collator for the moves sequence padding. It is padded to maximum possible length
    of the moves sequences to ensure fixed-size batches.
    """
    def __init__(self, max_lookahead: int):
        self.max_lookahead = max_lookahead

    def __call__(self, batch):
        tokens, tokens_, legal_moves, attacks, legal_moves_, attacks_, moves_list, lengths = zip(*batch)

        tokens = torch.stack(tokens)
        tokens_ = torch.stack(tokens_)

        legal_moves = torch.stack(legal_moves)
        attacks = torch.stack(attacks)

        legal_moves_ = torch.stack(legal_moves_)
        attacks_ = torch.stack(attacks_)

        batch_size = len(batch)
        max_len = 1 + self.max_lookahead

        moves = torch.zeros(
            (batch_size, max_len, SSLConstants.NUM_TOKENS_PER_MOVE),
            dtype=torch.long
        )
        for i, move_seq in enumerate(moves_list):
            seq_len = move_seq.shape[0]
            if seq_len > 0:
                moves[i, :seq_len, :] = move_seq

        lengths = torch.tensor(lengths, dtype=torch.long)

        moves_attention_mask = (torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)).long()
        moves_attention_mask = moves_attention_mask.unsqueeze(-1).repeat(
            1, 1, SSLConstants.NUM_TOKENS_PER_MOVE
        ).view(batch_size, -1)

        return tokens, tokens_, legal_moves, attacks, legal_moves_, attacks_, moves, moves_attention_mask


def encode_move_for_predictor(
        move: chess.Move,
        piece_type: chess.PieceType,
        taken_piece_type: chess.PieceType | None,
        turn: chess.Color,
        perspective: chess.Color):
    """Encodes a move in six tokens for the predictor:
    1. initial square
    2. target square
    3. relative color of the player (us/them)
    4. piece type moved
    5. piece type taken (0 if none), including pawn for en passant
    6. piece type promoted to (0 if none)
    """
    from_square = move.from_square
    to_square = move.to_square

    if perspective == chess.BLACK:
        from_square = chess.square_mirror(from_square)
        to_square = chess.square_mirror(to_square)

    relative_turn = int(turn == perspective)

    if taken_piece_type is None:
        taken_piece_type = 0
    if move.promotion is None:
        promotion_piece_type = 0
    else:
        promotion_piece_type = move.promotion

    move_encoded = torch.as_tensor(
        [
            from_square, to_square, relative_turn, piece_type, taken_piece_type, promotion_piece_type
        ],
        dtype=torch.long
    )
    return move_encoded


def get_relative_attack_map(board: chess.Board):
    """Relative attack map: number of attackers (control) we have
    over a square minus number of attackers they have.
    """
    if board.turn == chess.BLACK:
        board = board.mirror()

    attack_map = torch.zeros(64)
    for square in chess.SQUARES:
        attack_map[square] += len(board.attackers(chess.WHITE, square))
        attack_map[square] -= len(board.attackers(chess.BLACK, square))
    attack_map.clamp_(-ChessConstants.RELEVANT_ATTACKERS, ChessConstants.RELEVANT_ATTACKERS)
    attack_map += ChessConstants.RELEVANT_ATTACKERS
    return attack_map.long()

def encode_both_boards(
        board: chess.Board,
        encoding: str,
        min_moves: int,
        move_idx: int,
        target_idx: int,
        movelist: list[str]
    ):
    legal_moves = torch.zeros(ChessConstants.NUM_POLICY_CLASSES, dtype=torch.bool)
    for move in board.legal_moves:
        legal_moves[get_move_id(move, board.turn)] = True

    tokens = encode_board(board, encoding)
    attacks = get_relative_attack_map(board)
    movelist_ = torch.zeros((target_idx - move_idx, SSLConstants.NUM_TOKENS_PER_MOVE), dtype=torch.long)

    _board = board.copy(stack=False)
    for k in range(min_moves+move_idx, min_moves+target_idx):
        move = chess.Move.from_uci(movelist[k])

        moved_piece_type = _board.piece_at(move.from_square).piece_type

        if _board.piece_at(move.to_square) is None:
            if move.to_square == _board.ep_square and moved_piece_type == chess.PAWN:
                taken_piece_type = chess.PAWN
            else:
                taken_piece_type = None
        else:
            if _board.piece_at(move.to_square).color == _board.turn:
                taken_piece_type = None
            else:
                taken_piece_type = _board.piece_at(move.to_square).piece_type

        movelist_[k - (min_moves+move_idx), :] = encode_move_for_predictor(
            move=move,
            piece_type=moved_piece_type,
            taken_piece_type=taken_piece_type,
            turn=_board.turn,
            perspective=board.turn
        )
        _board.push(move)

    tokens_ = encode_board(_board, encoding)
    attacks_ = get_relative_attack_map(_board)
    legal_moves_ = torch.zeros(ChessConstants.NUM_POLICY_CLASSES, dtype=torch.bool)
    for move in _board.legal_moves:
        legal_moves_[get_move_id(move, _board.turn)] = True

    return tokens, tokens_, legal_moves, attacks, legal_moves_, attacks_, movelist_, target_idx - move_idx