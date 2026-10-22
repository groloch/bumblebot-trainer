import multiprocessing as mp
import os
import queue
import random
import shutil
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm
import torch

from ..config import StockfishEvaluationConfig, SSLModelConfig
from ..modeling import ChessModel, SSLChessModel, PositionClassificationHeads
from ..modeling.utils import load_teacher, load_model_state_dict
from .onnx_policy import OnnxPolicyModel
from .stockfish_game_worker import play_game_pair
from ..utils import build_model_config, load_config_file
from .bayes_elo import bayes_elo
from .consortium import PolicyConsortium
from .utils import (
    atomic_write_json,
    autocast,
    get_model_embeddings,
    load_run_config,
    resolve_checkpoint,
    resolve_path,
    stack_positions,
    load_openings
)


def _load_model_from_logdir(
    logdir: Path,
    checkpoint_path: str | None,
):
    config = load_run_config(logdir)
    training_name = config.get('training', {}).get('name')
    if not training_name:
        raise ValueError(f'Missing training.name in run config under {logdir}')
    checkpoint = resolve_checkpoint(logdir, training_name, checkpoint_path)

    encoding = config.get('data', {}).get('encoding')
    if not encoding:
        raise ValueError(f'Missing data.encoding in run config under {logdir}')

    model_config = build_model_config(config['model'])
    run_type = config.get('type')
    if run_type == 'ssl':
        ssl_config = SSLModelConfig(**vars(model_config))
        model = SSLChessModel(ssl_config)
    elif run_type in ('pv', 'pv_finetune', 'pvtuner'):
        model = ChessModel(model_config)
    else:
        raise ValueError(f'Unsupported model type for Stockfish evaluation: {run_type!r}')

    state_dict = torch.load(checkpoint, map_location='cpu', weights_only=True)
    load_teacher(model, state_dict)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    return model, encoding, config, checkpoint, device


def _load_consortium(
    consortium_config: dict,
    config_dir: Path,
):
    groups = consortium_config.get('expert_groups', [])
    supported_groups = {'phase', 'activity', 'evaluation'}
    if not groups or len(groups) != len(set(groups)) or not set(groups) <= supported_groups:
        raise ValueError(
            f'expert_groups must be a non-empty unique subset of {sorted(supported_groups)}'
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    router_base = None
    router_heads = None
    checkpoint_info = {'experts': {}}
    encoding = None

    if 'activity' in groups or 'evaluation' in groups:
        router_config = consortium_config.get('router')
        if router_config is None:
            raise ValueError('Router paths are required for activity or evaluation groups')

        classifier_logdir = resolve_path(router_config['classification_logdir'], config_dir)
        classifier_config = load_run_config(classifier_logdir)
        if classifier_config.get('type') != 'classification':
            raise ValueError('The router run must use type: classification')

        router_training = classifier_config.get('training', {})
        classifier_checkpoint = router_config.get('classification_checkpoint')
        if classifier_checkpoint is not None:
            classifier_checkpoint = str(resolve_path(classifier_checkpoint, classifier_logdir))
        classifier_checkpoint = resolve_checkpoint(
            classifier_logdir,
            router_training['name'],
            classifier_checkpoint
        )

        base_logdir = resolve_path(router_config['base_model_logdir'], config_dir)
        base_config = load_run_config(base_logdir)
        if base_config.get('type') != 'ssl':
            raise ValueError('The router base model must use type: ssl')

        base_training = base_config.get('training', {})
        base_checkpoint = router_config.get('base_model_checkpoint')
        if base_checkpoint is not None:
            base_checkpoint = str(resolve_path(base_checkpoint, base_logdir))
        base_checkpoint = resolve_checkpoint(
            base_logdir,
            base_training['name'],
            base_checkpoint
        )

        router_encoding = classifier_config['data']['encoding']
        if base_config['data']['encoding'] != router_encoding:
            raise ValueError('The router and its base model must use the same board encoding')
        encoding = router_encoding

        model_config = build_model_config(classifier_config['model'])
        router_base = SSLChessModel(SSLModelConfig(**vars(model_config)))
        load_model_state_dict(
            router_base,
            torch.load(base_checkpoint, map_location='cpu', weights_only=True),
        )
        router_base.to(device).eval()

        router_heads = PositionClassificationHeads(model_config.hidden_size)
        router_heads.load_state_dict(
            torch.load(classifier_checkpoint, map_location='cpu', weights_only=True)
        )
        router_heads.to(device).eval()
        checkpoint_info['router_base'] = str(base_checkpoint)
        checkpoint_info['router_heads'] = str(classifier_checkpoint)

    expected_experts = {
        'phase': ('opening', 'middlegame', 'endgame'),
        'activity': ('strategical', 'tactical'),
        'evaluation': ('winning', 'drawing', 'losing'),
    }
    experts = {}
    for group in groups:
        group_experts = consortium_config.get('experts', {}).get(group, {})
        missing = set(expected_experts[group]) - set(group_experts)
        if missing:
            raise ValueError(f'Missing {group} experts: {sorted(missing)}')

        experts[group] = {}
        checkpoint_info['experts'][group] = {}
        for expert_name in expected_experts[group]:
            expert_spec = group_experts[expert_name]
            if isinstance(expert_spec, str):
                expert_logdir = resolve_path(expert_spec, config_dir)
                expert_checkpoint = None
            else:
                expert_logdir = resolve_path(expert_spec['logdir'], config_dir)
                expert_checkpoint = expert_spec.get('checkpoint')
                if expert_checkpoint is not None:
                    expert_checkpoint = str(resolve_path(expert_checkpoint, expert_logdir))

            model, expert_encoding, _, checkpoint, expert_device = _load_model_from_logdir(
                expert_logdir,
                expert_checkpoint
            )
            if encoding is None:
                encoding = expert_encoding
            elif encoding != expert_encoding:
                raise ValueError(
                    f'Expert {group}/{expert_name} uses encoding {expert_encoding!r}; '
                    f'expected {encoding!r}'
                )
            if expert_device != device:
                model.to(device)
                model.eval()
            experts[group][expert_name] = model
            checkpoint_info['experts'][group][expert_name] = str(checkpoint)

    return (
        PolicyConsortium(router_base, router_heads, experts, groups),
        encoding,
        checkpoint_info,
        device,
    )


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
    if isinstance(model, PolicyConsortium):
        return model.predict_move_ids(batch, device)

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

    tokens, extra = stack_positions(batch, device)
    with torch.inference_mode(), autocast(device):
        squares_embeddings, _ = get_model_embeddings(model, tokens, extra)
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
    book_path: str | None = None,
) -> dict:
    evaluation_config_path = Path(evaluation_config_path).expanduser().resolve()
    raw_evaluation_config = load_config_file(str(evaluation_config_path))
    evaluation_config = StockfishEvaluationConfig(**raw_evaluation_config)
    _validate_config(evaluation_config)

    model_sources = sum((logdir is not None, onnx_checkpoint_path is not None,
                         evaluation_config.consortium is not None))
    if model_sources != 1:
        raise ValueError('Choose exactly one model source: --logdir, --onnx, or consortium config')
    if onnx_checkpoint_path is not None and checkpoint_path is not None:
        raise ValueError('--onnx cannot be combined with a logdir checkpoint override')
    if evaluation_config.consortium is not None and checkpoint_path is not None:
        raise ValueError('Consortium checkpoints are specified in the evaluation config')

    logdir_path = None
    if logdir is not None:
        logdir_path = Path(logdir).expanduser().resolve()
        if not logdir_path.is_dir():
            raise NotADirectoryError(f'Model logdir not found: {logdir_path}')

    config_dir = evaluation_config_path.parent
    if book_path is not None:
        resolved_book_path = resolve_path(book_path, Path.cwd())
    elif evaluation_config.book_path is not None:
        resolved_book_path = resolve_path(evaluation_config.book_path, config_dir)
    else:
        resolved_book_path = Path('book.csv').expanduser().resolve()
    openings = load_openings(resolved_book_path)

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
    elif evaluation_config.consortium is not None:
        model, encoding, checkpoint, device = _load_consortium(
            evaluation_config.consortium,
            config_dir
        )
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
    model_source = (
        'standalone_onnx' if onnx_checkpoint_path is not None else
        'consortium' if evaluation_config.consortium is not None else
        'logdir'
    )
    summary = {
        'model_source': model_source,
        'checkpoint': str(checkpoint) if isinstance(checkpoint, Path) else checkpoint,
        'expert_groups': (
            evaluation_config.consortium['expert_groups']
            if evaluation_config.consortium is not None else None
        ),
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

    results_path = (
        resolve_path(evaluation_config.results_path, config_dir)
        if evaluation_config.results_path else None
    )
    if results_path is None:
        if logdir_path is not None:
            results_dir = logdir_path
        elif evaluation_config.consortium is not None:
            results_dir = config_dir
        else:
            results_dir = checkpoint.parent
        results_path = results_dir / 'stockfish_eval_results.json'
    atomic_write_json(results_path, output)

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
