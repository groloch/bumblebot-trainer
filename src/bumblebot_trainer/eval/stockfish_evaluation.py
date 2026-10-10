import csv
import json
import multiprocessing as mp
import os
import queue
import random
import shutil
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ..config import StockfishEvaluationConfig
from .onnx_policy import OnnxPolicyModel
from .stockfish_game_worker import play_game_pair
from ..utils import build_model_config, load_config_file
from .bayes_elo import bayes_elo


def _resolve_checkpoint(logdir: Path, training_name: str, explicit_path: str | None) -> Path:
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

    candidates = []
    prefix = f'{training_name}_step'
    for candidate in logdir.glob(f'{prefix}*.pth'):
        step = candidate.stem[len(prefix):]
        if step.isdigit():
            candidates.append((int(step), candidate))
    if candidates:
        return max(candidates, key=lambda item: item[0])[1]

    raise FileNotFoundError(
        f'Could not find {training_name}.pth or a {training_name}_step*.pth checkpoint in {logdir}'
    )


def _load_model_from_logdir(
    logdir: Path,
    checkpoint_path: str | None,
):
    config_paths = sorted(logdir.glob('*.yaml')) + sorted(logdir.glob('*.yml'))
    if not config_paths:
        raise FileNotFoundError(f'No config YAML file found in {logdir}')
    if len(config_paths) != 1:
        raise RuntimeError(
            f'Expected one config YAML file in {logdir}, found: '
            + ', '.join(path.name for path in config_paths)
        )

    # Reuse the project config helpers; leave the source config and logdir untouched.
    config = load_config_file(str(config_paths[0]))
    training_name = config.get('training', {}).get('name')
    if not training_name:
        raise ValueError(f'Missing training.name in {config_paths[0]}')
    checkpoint = _resolve_checkpoint(logdir, training_name, checkpoint_path)

    encoding = config.get('data', {}).get('encoding')
    if not encoding:
        raise ValueError(f'Missing data.encoding in {config_paths[0]}')

    import torch

    from ..config import SSLModelConfig
    from ..modeling import ChessModel, SSLChessModel

    model_config = build_model_config(config['model'])
    run_type = config.get('type')
    if run_type == 'ssl':
        ssl_config = SSLModelConfig(**vars(model_config))
        model = SSLChessModel(ssl_config)
    elif run_type in ('pv', 'pvtuner'):
        model = ChessModel(model_config)
    else:
        raise ValueError(f'Unsupported model type for Stockfish evaluation: {run_type!r}')

    state_dict = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    return model, encoding, config, checkpoint, device


def _load_standalone_onnx(
    checkpoint_path: str,
    evaluation_config: StockfishEvaluationConfig,
):
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if checkpoint.suffix.lower() != '.onnx':
        raise ValueError(f'Standalone checkpoint must be an .onnx file: {checkpoint}')
    if not checkpoint.is_file():
        raise FileNotFoundError(f'ONNX checkpoint not found: {checkpoint}')

    encoding = evaluation_config.encoding
    input_sizes = {'simplified': 18, 'legal': 82, 'lc0': 112}
    if encoding not in input_sizes:
        raise ValueError(
            'Standalone ONNX mode requires evaluation config field encoding to be '
            f'one of {sorted(input_sizes)}; got {encoding!r}'
        )
    model = OnnxPolicyModel(
        checkpoint,
        input_size=input_sizes[encoding],
        tokens_input=evaluation_config.onnx_tokens_input,
        legal_input=evaluation_config.onnx_legal_input,
        policy_output=evaluation_config.onnx_policy_output,
    )
    print(f'Loaded standalone ONNX policy from {checkpoint} using {model.providers}')
    return model, encoding, checkpoint


def _load_openings(book_path: Path) -> list[str]:
    if not book_path.is_file():
        raise FileNotFoundError(f'Opening book not found: {book_path}')

    openings = []
    with book_path.open(newline='') as book_file:
        for row in csv.reader(book_file, delimiter=';'):
            if row and row[0].strip():
                openings.append(row[0].strip())
    if not openings:
        raise ValueError(f'No starting positions found in opening book {book_path}')
    return openings


def _sample_stockfish_elo(rng: random.Random, ranges: list[list[int]]) -> int:
    if not ranges:
        raise ValueError('stockfish_elo_ranges must contain at least one range')
    bounds = rng.choice(ranges)
    if len(bounds) != 2:
        raise ValueError(f'Each Stockfish Elo range must be [min, max], got {bounds!r}')
    low, high = map(int, bounds)
    if low > high:
        raise ValueError(f'Invalid Stockfish Elo range [{low}, {high}]')
    return rng.randint(low, high)


def _evaluate_batch(model, device, batch: list[dict]) -> list[int]:
    if isinstance(model, OnnxPolicyModel):
        tokens = np.stack([item['tokens'] for item in batch])
        extra_keys = set(batch[0]['extra'])
        extra = {
            key: np.stack([item['extra'][key] for item in batch])
            for key in extra_keys
        }
        logits = model.predict(tokens, extra)
        chosen = []
        for row, item in enumerate(batch):
            legal_ids = np.asarray(item['legal_ids'], dtype=np.int64)
            if legal_ids.size == 0 or legal_ids.max() >= logits.shape[1]:
                raise ValueError('ONNX policy output does not cover the legal move ids')
            chosen.append(int(legal_ids[np.argmax(logits[row, legal_ids])]))
        return chosen

    import torch

    tokens = torch.from_numpy(np.stack([item['tokens'] for item in batch])).to(device)
    extra_keys = set(batch[0]['extra'])
    extra = {
        key: torch.from_numpy(np.stack([item['extra'][key] for item in batch])).to(device)
        for key in extra_keys
    }

    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == 'cuda',
    ):
        embedded = model.embed(tokens, **extra)
        if hasattr(embedded, 'squares_embeddings'):
            squares_embeddings = embedded.squares_embeddings
        elif isinstance(embedded, tuple):
            squares_embeddings = embedded[0]
        else:
            raise TypeError(f'Unsupported model embed output: {type(embedded).__name__}')
        logits = model.policy_head(squares_embeddings, None).logits

    chosen = []
    for row, item in enumerate(batch):
        legal_ids = torch.as_tensor(item['legal_ids'], dtype=torch.long, device=logits.device)
        best = int(legal_ids[logits[row, legal_ids].argmax()].item())
        chosen.append(best)
    return chosen


def _run_inference_worker(
    model,
    device,
    config: StockfishEvaluationConfig,
    request_queue,
    response_queues,
    processes,
    result_queue,
    pair_ids: list[int],
    progress,
):
    results = {}
    dead_since = {}
    pair_id_set = set(pair_ids)

    while len(results) < len(pair_ids):
        batch = []
        try:
            first_request = request_queue.get(timeout=0.02)
            batch.append(first_request)
        except queue.Empty:
            pass

        if batch:
            deadline = time.monotonic() + max(config.batch_timeout_ms, 0.0) / 1000.0
            while len(batch) < config.batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    batch.append(request_queue.get(timeout=remaining))
                except queue.Empty:
                    break

            try:
                move_ids = _evaluate_batch(model, device, batch)
                for request, move_id in zip(batch, move_ids):
                    response_queues[request['game_id']].put({'move_id': move_id})
            except BaseException as exc:
                message = f'{type(exc).__name__}: {exc}'
                for request in batch:
                    response_queues[request['game_id']].put({'error': message})

        while True:
            try:
                event = result_queue.get_nowait()
            except queue.Empty:
                break
            pair_id = event['pair_id']
            if pair_id in pair_id_set and pair_id not in results:
                results[pair_id] = event
                progress.update(1)

        # Surface a hard worker crash even if it could not send its error event.
        for pair_id, process in processes.items():
            if pair_id in results or process.exitcode is None:
                continue
            dead_at = dead_since.setdefault(pair_id, time.monotonic())
            if time.monotonic() - dead_at > 1.0:
                results[pair_id] = {
                    'pair_id': pair_id,
                    'error': f'Game worker exited with code {process.exitcode} without a result',
                }
                progress.update(1)

    return [results[pair_id] for pair_id in pair_ids]


def _validate_config(config: StockfishEvaluationConfig):
    if config.num_game_pairs <= 0:
        raise ValueError('num_game_pairs must be positive')
    if config.num_workers <= 0:
        raise ValueError('num_workers must be positive')
    if config.batch_size <= 0:
        raise ValueError('batch_size must be positive')
    if config.move_time_seconds <= 0:
        raise ValueError('move_time_seconds must be positive')
    if config.max_game_plies <= 0:
        raise ValueError('max_game_plies must be positive')
    if config.elo_step <= 0 or config.elo_min > config.elo_max:
        raise ValueError('Invalid Elo estimation grid')
    for bounds in config.stockfish_elo_ranges:
        if len(bounds) != 2 or int(bounds[0]) > int(bounds[1]):
            raise ValueError(f'Invalid Stockfish Elo range: {bounds!r}')


def run_stockfish_evaluation(
    evaluation_config_path: str,
    *,
    logdir: str | None = None,
    checkpoint_path: str | None = None,
    onnx_checkpoint_path: str | None = None,
) -> dict:
    if (logdir is None) == (onnx_checkpoint_path is None):
        raise ValueError('Choose exactly one model source: --logdir or --onnx')
    if onnx_checkpoint_path is not None and checkpoint_path is not None:
        raise ValueError('--onnx cannot be combined with a logdir checkpoint override')

    logdir_path = None
    if logdir is not None:
        logdir_path = Path(logdir).expanduser().resolve()
        if not logdir_path.is_dir():
            raise NotADirectoryError(f'Model logdir not found: {logdir_path}')

    evaluation_config_path = Path(evaluation_config_path).expanduser().resolve()
    raw_evaluation_config = load_config_file(str(evaluation_config_path))
    evaluation_config = StockfishEvaluationConfig(**raw_evaluation_config)
    _validate_config(evaluation_config)

    config_dir = evaluation_config_path.parent
    book_path = Path(evaluation_config.book_path).expanduser()
    if not book_path.is_absolute():
        book_path = (config_dir / book_path).resolve()
    openings = _load_openings(book_path)

    stockfish_path = evaluation_config.stockfish_path
    if '/' in stockfish_path or os.sep in stockfish_path:
        stockfish_path = str((config_dir / stockfish_path).expanduser().resolve())
    elif shutil.which(stockfish_path) is None:
        raise FileNotFoundError(
            f'Stockfish executable {stockfish_path!r} was not found on PATH; '
            'set stockfish_path in the evaluation config'
        )

    if onnx_checkpoint_path is not None:
        model, encoding, checkpoint = _load_standalone_onnx(
            onnx_checkpoint_path, evaluation_config
        )
        device = None
    else:
        model, encoding, _, checkpoint, device = _load_model_from_logdir(
            logdir_path, checkpoint_path
        )

    rng = random.Random(evaluation_config.seed)
    pair_specs = [
        (rng.choice(openings), _sample_stockfish_elo(rng, evaluation_config.stockfish_elo_ranges))
        for _ in range(evaluation_config.num_game_pairs)
    ]

    context = mp.get_context('spawn')
    request_queue = context.Queue()
    result_queue = context.Queue()
    pair_results = []
    progress = tqdm(total=evaluation_config.num_game_pairs, desc='Stockfish game pairs')
    try:
        # Run pairs in waves so only num_workers pair subprocesses (and their
        # two games each) can be active at a time.
        for offset in range(0, evaluation_config.num_game_pairs, evaluation_config.num_workers):
            pair_ids = list(range(
                offset,
                min(offset + evaluation_config.num_workers, evaluation_config.num_game_pairs),
            ))
            processes = {}
            response_queues = {
                game_id: context.Queue()
                for pair_id in pair_ids
                for game_id in (pair_id * 2, pair_id * 2 + 1)
            }
            try:
                for pair_id in pair_ids:
                    start_fen, stockfish_elo = pair_specs[pair_id]
                    process = context.Process(
                        target=play_game_pair,
                        args=(
                            pair_id,
                            start_fen,
                            stockfish_elo,
                            encoding,
                            stockfish_path,
                            evaluation_config.move_time_seconds,
                            evaluation_config.max_game_plies,
                            request_queue,
                            (response_queues[pair_id * 2], response_queues[pair_id * 2 + 1]),
                            result_queue,
                        ),
                        name=f'stockfish-game-pair-{pair_id}',
                    )
                    process.start()
                    processes[pair_id] = process

                pair_results.extend(_run_inference_worker(
                    model,
                    device,
                    evaluation_config,
                    request_queue,
                    response_queues,
                    processes,
                    result_queue,
                    pair_ids,
                    progress,
                ))
            finally:
                for process in processes.values():
                    process.join(timeout=5.0)
                    if process.is_alive():
                        process.terminate()
                        process.join()
                for channel in response_queues.values():
                    channel.close()
                    channel.join_thread()
    finally:
        progress.close()
        for channel in (request_queue, result_queue):
            channel.close()
            channel.join_thread()

    successful_pairs = [pair for pair in pair_results if 'games' in pair]
    game_results = [game for pair in successful_pairs for game in pair['games']]
    opponent_elos = [pair['stockfish_elo'] for pair in successful_pairs for _ in pair['games']]
    outcomes = [game['score'] for game in game_results]
    elo, elo_std = bayes_elo(
        opponent_elos,
        outcomes,
        prior_mean=evaluation_config.prior_elo_mean,
        prior_std=evaluation_config.prior_elo_std,
        elo_min=evaluation_config.elo_min,
        elo_max=evaluation_config.elo_max,
        elo_step=evaluation_config.elo_step,
    )

    wins = sum(score == 1.0 for score in outcomes)
    draws = sum(score == 0.5 for score in outcomes)
    losses = sum(score == 0.0 for score in outcomes)
    summary = {
        'model_source': 'standalone_onnx' if onnx_checkpoint_path is not None else 'logdir',
        'checkpoint': str(checkpoint),
        'num_game_pairs': len(successful_pairs),
        'num_games': len(game_results),
        'wins': int(wins),
        'draws': int(draws),
        'losses': int(losses),
        'score': float(np.mean(outcomes)) if outcomes else 0.0,
        'elo': elo,
        'elo_std': elo_std,
    }
    output = {
        'summary': summary,
        'pairs': pair_results,
    }

    results_path = evaluation_config.results_path
    if results_path:
        results_path = Path(results_path).expanduser()
        if not results_path.is_absolute():
            results_path = (config_dir / results_path).resolve()
    else:
        results_dir = logdir_path if logdir_path is not None else checkpoint.parent
        results_path = results_dir / 'stockfish_eval_results.json'
    results_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = results_path.with_suffix(results_path.suffix + '.tmp')
    with temporary_path.open('w') as output_file:
        json.dump(output, output_file, indent=2)
        output_file.write('\n')
    os.replace(temporary_path, results_path)

    print(
        f'Stockfish evaluation: Elo {elo:.0f} ± {elo_std:.0f}; '
        f'{wins} wins, {draws} draws, {losses} losses '
        f'({summary["score"]:.1%} score). Results: {results_path}'
    )
    errors = [pair for pair in pair_results if 'error' in pair]
    if errors:
        raise RuntimeError(
            f'{len(errors)} game pair(s) failed; partial results saved to {results_path}. '
            f'First failure: {errors[0]["error"]}'
        )
    return output
