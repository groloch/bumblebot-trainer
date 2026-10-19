from copy import deepcopy
import yaml

import numpy as np
import torch
import chess


from .config import ModelConfig, EncoderConfig, CFEncoderConfig


class ChessConstants:
    NUM_POLICY_CLASSES = 4288 # 64x64 + 8x8x3 for promotions
    CONTEXT_LENGTH = 64 # 64 squares
    MAX_NUMBER_OF_MOVES = 256

    MAX_ATTACKERS = 16
    RELEVANT_ATTACKERS = 4

    NUM_PIECE_TYPES = 6
    NUM_COLORS = 2
    NUM_SQUARES = 64


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
    """Converts an expected result in [0, 1] to centipawns using Lichess' curve."""
    if winpercent <= 0:
        return -1000
    elif winpercent >= 1:
        return 1000
    cp = np.log(winpercent / (1 - winpercent)) / magic
    cp = max(min(cp, 1000), -1000)
    return int(cp)


def wdl_to_expected_result(
    wdl: torch.Tensor,
    *,
    from_logits: bool = False,
) -> torch.Tensor:
    """Convert W/D/L model outputs to expected score (win + 0.5 * draw).

    Cross-entropy heads produce logits; regression heads produce probability-like
    values, which are clamped and renormalized. Scalar expected-result outputs are
    passed through unchanged (apart from clipping to [0, 1]).
    """
    values = wdl.float()
    if values.ndim == 0:
        return values.clamp(0.0, 1.0)
    if values.shape[-1] == 1:
        return values.squeeze(-1).clamp(0.0, 1.0)
    if values.shape[-1] != 3:
        raise ValueError(f'Expected a scalar or W/D/L output, got shape {tuple(values.shape)}')

    if from_logits:
        probabilities = torch.softmax(values, dim=-1)
    else:
        probabilities = values.clamp_min(0.0)
        mass = probabilities.sum(dim=-1, keepdim=True)
        fallback = torch.softmax(values, dim=-1)
        probabilities = torch.where(
            mass > 1e-8,
            probabilities / mass.clamp_min(1e-8),
            fallback,
        )
    return probabilities[..., 0] + 0.5 * probabilities[..., 1]


def expected_result_to_cp(
    expected_result: torch.Tensor,
    magic: float = 0.00368208,
    max_cp: float = 1000.0,
) -> torch.Tensor:
    """Convert expected score in [0, 1] to centipawns via Lichess' curve."""
    probability = expected_result.float().clamp(1e-6, 1.0 - 1e-6)
    return (torch.logit(probability) / magic).clamp(-max_cp, max_cp)

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

def normalize_config(config: dict) -> dict:
    ENCODING_TO_INPUT_SIZE = {
        'lc0': 112,
        'simplified': 18,
        'legal': 18+64
    }

    config = deepcopy(config)

    if 'training' in config:
        # yaml doesn't always keep scientific notation values as desired in this codebase
        config['training']['learning_rate'] = float(config['training']['learning_rate'])

        encoding = config.get('data', {}).get('encoding')
        if encoding in ENCODING_TO_INPUT_SIZE:
            config['model']['input_size'] = ENCODING_TO_INPUT_SIZE[encoding]

        if 'model' in config and 'encoder' in config['model']:
            config['model']['intermediate_size'] = config['model']['encoder']['intermediate_size']

    return config


def load_config_file(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return normalize_config(config)


def build_model_config(model_config_dict: dict) -> ModelConfig:
    model_dict = deepcopy(model_config_dict)
    hidden_size = model_dict['hidden_size']
    encoder_name = model_dict['encoder_name']
    encoder_kwargs = model_dict.pop('encoder')

    if encoder_name in ('cf', 'cfc'):
        encoder_config = CFEncoderConfig(hidden_size=hidden_size, **encoder_kwargs)
    else:
        encoder_config = EncoderConfig(hidden_size=hidden_size, **encoder_kwargs)

    return ModelConfig(
        encoder_config=encoder_config,
        **model_dict,
    )
