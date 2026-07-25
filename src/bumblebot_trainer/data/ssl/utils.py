import torch
import chess


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
