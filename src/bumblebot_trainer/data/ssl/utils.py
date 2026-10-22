import torch
import chess

from ..utils import encode_board, VariationNode, process_item


class SSLConstants:
    NUM_TOKENS_PER_MOVE = 6


def collate_supervision_targets(targets):
    """Collate the policy and value labels together with their source-row indices.
    """
    output = {}

    for name in ('policy', 'value', 'qvalue'):
        mask_name = f'{name}_mask'
        valid = [bool(target.get(mask_name, True)) for target in targets]
        indices = torch.tensor(
            [i for i, is_valid in enumerate(valid) if is_valid],
            dtype=torch.long,
        )
        selected_targets = [
            target[name] for target, is_valid in zip(targets, valid) if is_valid
        ]
        if selected_targets:
            output[name] = torch.stack(selected_targets)
        else:
            output[name] = targets[0][name].new_empty(
                (0, *targets[0][name].shape)
            )
        output[f'valid_{name}_indices'] = indices

    return output


class PositionCollator:
    """Collate position batches while retaining indices for available labels."""
    def __call__(self, batch):
        tokens, targets, extra = zip(*batch)

        tokens = torch.stack(tokens)
        targets = collate_supervision_targets(targets)
        keys = extra[0].keys()
        extra = {key: torch.stack([item[key] for item in extra]) for key in keys}

        return tokens, targets, extra


class SSLCollator:
    """Collate SSL examples and retain indices for each available supervision target."""
    def __init__(self, max_lookahead: int):
        self.max_lookahead = max_lookahead

    def __call__(self, batch):
        tokens, tokens_, moves_list, lengths, target_samples, extra, extra_ = zip(*batch)

        tokens = torch.stack(tokens)
        targets = collate_supervision_targets(target_samples)
        valid_ssl_indices = torch.tensor(
            [
                i for i, target in enumerate(target_samples)
                if bool(target.get('ssl_mask', True))
            ],
            dtype=torch.long,
        )
        targets['valid_ssl_indices'] = valid_ssl_indices

        selected_ssl_indices = valid_ssl_indices.tolist()
        if selected_ssl_indices:
            tokens_ = torch.stack([tokens_[i] for i in selected_ssl_indices])
        else:
            tokens_ = tokens_[0].new_empty((0, *tokens_[0].shape))
        moves_list = [moves_list[i] for i in selected_ssl_indices]
        lengths = torch.tensor(
            [lengths[i] for i in selected_ssl_indices],
            dtype=torch.long,
        )

        batch_size = len(moves_list)
        max_len = max((sequence.shape[0] for sequence in moves_list), default=1)
        moves = torch.zeros(
            (batch_size, max_len, SSLConstants.NUM_TOKENS_PER_MOVE),
            dtype=torch.long,
        )
        for i, move_seq in enumerate(moves_list):
            seq_len = move_seq.shape[0]
            if seq_len > 0:
                moves[i, :seq_len, :] = move_seq

        moves_attention_mask = (torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)).long()
        moves_attention_mask = moves_attention_mask.unsqueeze(-1).repeat(
            1, 1, SSLConstants.NUM_TOKENS_PER_MOVE
        ).view(batch_size, max_len * SSLConstants.NUM_TOKENS_PER_MOVE)

        keys = extra[0].keys()
        extra = {key: torch.stack([item[key] for item in extra]) for key in keys}
        extra_ = {}
        for key in keys:
            selected_values = [extra_[i][key] for i in selected_ssl_indices]
            if selected_values:
                extra_[key] = torch.stack(selected_values)
            else:
                extra_[key] = extra_[0][key].new_empty(
                    (0, *extra_[0][key].shape)
                )

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

def encode_move_sequence(
        board: chess.Board,
        movelist: list[str],
        start_idx: int,
        end_idx: int
    ):
    """Encodes a sequence of moves in uci format into a predictor input (6 tokens per move). See
    encode_move_for_predictor for more info on the encoding used for each moves.
    """
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
        value_: float | torch.Tensor | None = None,
        qvalue: float | torch.Tensor | None = None,
        qvalue_: float | torch.Tensor | None = None,
        policy_valid: bool | None = None,
        value_valid: bool | None = None,
        qvalue_valid: bool | None = None,
        ssl_valid: bool = True
    ):
    """Pre-processes a position and a list of moves for ssl training.
    Returns the encoded boards (current and after the move sequence), and the
    encoded move sequence (6 tokens per move). Also returns the policy and value targets.
    """
    tokens, targets, extra = process_item(
        board,
        nodes,
        encoding,
        value=value,
        qvalue=qvalue,
        policy_valid=policy_valid,
        value_valid=value_valid,
        qvalue_valid=qvalue_valid,
    )
    movelist_, board_ = encode_move_sequence(
        board=board,
        movelist=movelist,
        start_idx=min_moves + move_idx,
        end_idx=min_moves + target_idx
    )
    tokens_, targets_, extra_ = process_item(
        board_,
        nodes_,
        encoding,
        value=value_,
        qvalue=qvalue_,
        policy_valid=policy_valid,
        value_valid=value_valid,
        qvalue_valid=qvalue_valid,
    )

    targets['ssl_mask'] = torch.tensor(ssl_valid, dtype=torch.bool)
    targets['policy_'] = targets_['policy']
    targets['value_'] = targets_['value']
    targets['qvalue_'] = targets_['qvalue']

    return tokens, tokens_, movelist_, target_idx - move_idx, targets, extra, extra_
