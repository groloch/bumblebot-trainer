import json
import os
from functools import wraps
from pathlib import Path
import csv

import numpy as np
import chess.pgn
import torch
from torch.utils.data import DataLoader

from ..utils import load_config_file
from ..data.utils import encode_board


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
def preserve_model_mode(function):
    """Temporarily switch an evaluator's model to eval mode, then restore it."""
    @wraps(function)
    def wrapped(evaluator, *args, **kwargs):
        model = evaluator.model
        was_training = model.training
        model.eval()
        try:
            return function(evaluator, *args, **kwargs)
        finally:
            model.train(was_training)

    return wrapped


def resolve_path(path: str, base_dir: Path) -> Path:
    """Resolve a configured path relative to its owning directory."""
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = base_dir / resolved
    return resolved.resolve()


def resolve_checkpoint(logdir: Path, training_name: str, explicit_path: str | None) -> Path:
    """Resolve an explicit checkpoint or choose the latest checkpoint in a run."""
    if explicit_path is not None:
        checkpoint = Path(explicit_path).expanduser()
        if checkpoint.suffix.lower() == '.onnx':
            raise ValueError(
                'A standalone ONNX checkpoint cannot be combined with a model logdir; '
                'use the --onnx mode instead.'
            )
        if not checkpoint.is_file():
            raise FileNotFoundError(f'Checkpoint not found: {checkpoint}')
        return checkpoint

    final_checkpoint = logdir / f'{training_name}.pth'
    if final_checkpoint.is_file():
        return final_checkpoint

    prefix = f'{training_name}_step'
    candidates = []
    for candidate in logdir.glob(f'{prefix}*.pth'):
        step = candidate.stem[len(prefix):]
        if step.isdigit():
            candidates.append((int(step), candidate))
    if candidates:
        return max(candidates, key=lambda item: item[0])[1]

    raise FileNotFoundError(
        f'Could not find {training_name}.pth or a {training_name}_step*.pth '
        f'checkpoint in {logdir}'
    )


def load_run_config(logdir: Path) -> dict:
    """Load the sole YAML config stored in a training run directory."""

    config_paths = sorted(logdir.glob('*.yaml')) + sorted(logdir.glob('*.yml'))
    if not config_paths:
        raise FileNotFoundError(f'No config YAML file found in {logdir}')
    if len(config_paths) != 1:
        names = ', '.join(path.name for path in config_paths)
        raise RuntimeError(
            f'Expected one config YAML file in {logdir}, found: {names}'
        )
    return load_config_file(str(config_paths[0]))


def encode_position(board, encoding: str) -> dict:
    """Encode a board into the NumPy payload shared by evaluation workers."""

    encoded = encode_board(board, encoding)
    if isinstance(encoded, tuple):
        tokens, extra = encoded
    else:
        tokens, extra = encoded, {}
    return {
        'tokens': tokens.numpy(),
        'extra': {key: value.numpy() for key, value in extra.items()},
    }


def make_dataloader(dataset, config, device, collate_fn=None):
    """Build the consistently configured evaluation dataloader."""
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == 'cuda',
        drop_last=False,
        collate_fn=collate_fn,
    )


def stack_positions(positions, device):
    """Stack encoded positions into model-ready tensors on ``device``."""
    token_batch = np.stack([position['tokens'] for position in positions])
    tokens = torch.from_numpy(token_batch).to(device)
    keys = positions[0]['extra']
    extra = {
        key: torch.from_numpy(np.stack([position['extra'][key] for position in positions])).to(device)
        for key in keys
    }
    return tokens, extra


def get_model_embeddings(model, tokens, extra):
    """Return square and value/CLS embeddings across supported model outputs."""
    embedded = model.embed(tokens, **extra)
    if hasattr(embedded, 'squares_embeddings'):
        return embedded.squares_embeddings, embedded.cls_embedding
    if isinstance(embedded, tuple):
        squares = embedded[0]
        return squares, squares.mean(dim=1)
    raise TypeError(f'Unsupported model embed output: {type(embedded).__name__}')


def autocast(device):
    """Use bfloat16 autocast on CUDA and a no-op context on CPU."""
    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == 'cuda',
    )


def padded_legal_logits(logits, legal_ids):
    """Gather legal-move logits into a padded batch, masking padding entries."""
    lengths = [len(ids) for ids in legal_ids]
    if any(length == 0 for length in lengths):
        raise ValueError('A non-terminal position has no legal moves')

    max_moves = max(lengths)
    padded = np.zeros((len(legal_ids), max_moves), dtype=np.int64)
    mask = np.zeros((len(legal_ids), max_moves), dtype=np.bool_)
    for row, ids in enumerate(legal_ids):
        ids = np.asarray(ids, dtype=np.int64)
        if ids.max() >= logits.shape[-1]:
            raise ValueError('Policy output does not cover the legal move ids')
        padded[row, :len(ids)] = ids
        mask[row, :len(ids)] = True

    padded = torch.from_numpy(padded).to(logits.device)
    mask = torch.from_numpy(mask).to(logits.device)
    legal_logits = logits.float().gather(1, padded).masked_fill(~mask, float('-inf'))
    return padded, lengths, legal_logits


def atomic_write_json(path: Path, value: dict) -> None:
    """Write JSON atomically so interrupted evaluations do not corrupt results."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + '.tmp')
    with temporary_path.open('w', encoding='utf-8') as output_file:
        json.dump(value, output_file, indent=2)
        output_file.write('\n')
    os.replace(temporary_path, path)
