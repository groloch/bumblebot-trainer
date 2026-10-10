import torch
import chess

from ..utils import encode_board, VariationNode, process_item
from ...utils import ChessConstants

from typing import Literal


class SSLConstants:
    NUM_TOKENS_PER_MOVE = 6


class SSLCollator:
    """Custom collator for the moves sequence padding. It is padded to maximum possible length
    of the moves sequences to ensure fixed-size batches.
    """
    def __init__(self, max_lookahead: int):
        self.max_lookahead = max_lookahead

    def __call__(self, batch):
        tokens, tokens_, moves_list, lengths, targets, extra, extra_ = zip(*batch)

        tokens = torch.stack(tokens)
        tokens_ = torch.stack(tokens_)

        targets = {
            k: torch.stack([t[k] for t in targets]) for k in targets[0]
        }

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

        keys = extra[0].keys()
        extra = {k: torch.stack([e[k] for e in extra]) for k in keys}
        extra_ = {k: torch.stack([e_[k] for e_ in extra_]) for k in keys}

        return tokens, tokens_, targets, moves, moves_attention_mask, extra, extra_


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

def apply_td_value(
        color,
        advancement,
        result,
        smoothing: Literal['linear', 'quadratic', 'cubic'] = 'quadratic'):
    """Temporal difference formula applied to the game result to get the intermediate
    expected result at some move.
    It is a smoothing from draw expected (startpos) to the actual result of the game
    based on the advancement.

    Args:
        color (bool): current side to play
        advancement (float): current advancement of the game (moves played / total number of moves)
        result (int): result of the game (1: white wins, 0: draw, -1: black wins)
        smoothing (str): smoothing to apply

    Returns:
        float: expected result from the side to move perspective, in [0, 1]
    """
    prob = (result + 1) / 2

    def scale_a(a):
        match smoothing:
            case 'linear':
                return a
            case 'quadratic':
                return a ** 2
            case 'cubic':
                return a ** 3
    advancement_scaled = scale_a(advancement)

    value = 0.5 * (1 - advancement_scaled) + prob * advancement_scaled

    # process_item expects targets from the side-to-move perspective
    if color == chess.BLACK:
        value = 1.0 - value

    return value

def encode_move_sequence(
        board: chess.Board,
        movelist: list[str],
        start_idx: int,
        end_idx: int
    ):
    moves = torch.zeros((end_idx - start_idx, SSLConstants.NUM_TOKENS_PER_MOVE), dtype=torch.long)
    board_ = board.copy()

    for k in range(start_idx, end_idx):
        move = chess.Move.from_uci(movelist[k])
        moved_piece_type = board_.piece_at(move.from_square).piece_type

        if board_.piece_at(move.to_square) is None:
            if move.to_square == board_.ep_square and moved_piece_type == chess.PAWN:
                taken_piece_type = chess.PAWN
            else:
                taken_piece_type = None
        elif board_.piece_at(move.to_square).color == board_.turn:
            taken_piece_type = None
        else:
            taken_piece_type = board_.piece_at(move.to_square).piece_type

        moves[k - start_idx, :] = encode_move_for_predictor(
            move=move,
            piece_type=moved_piece_type,
            taken_piece_type=taken_piece_type,
            turn=board_.turn,
            perspective=board.turn
        )
        board_.push(move)

    return moves, board_


def encode_both_boards(
        board: chess.Board,
        nodes: list[VariationNode],
        nodes_: list[VariationNode],
        encoding: str,
        min_moves: int,
        move_idx: int,
        target_idx: int,
        movelist: list[str],
        value: float | torch.Tensor | None = None,
        value_: float | torch.Tensor | None = None
    ):
    tokens, targets, extra = process_item(board, nodes, encoding, value=value)
    movelist_, board_ = encode_move_sequence(
        board=board,
        movelist=movelist,
        start_idx=min_moves + move_idx,
        end_idx=min_moves + target_idx
    )
    tokens_, targets_, extra_ = process_item(board_, nodes_, encoding, value=value_)

    targets['policy_'] = targets_['policy']
    targets['value_'] = targets_['value']

    return tokens, tokens_, movelist_, target_idx - move_idx, targets, extra, extra_
