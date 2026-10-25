import itertools
from collections import OrderedDict
import multiprocessing as mp
import queue
import random
import time
from pathlib import Path

import chess
import chess.pgn

import numpy as np
from tqdm import tqdm
import torch

from ..modeling import PositionClassificationHeads, SSLChessModel
from ..modeling.utils import load_model_state_dict
from ..config import TournamentConfig, SSLModelConfig
from ..utils import build_model_config, load_config_file
from .bayes_elo import bayes_elo_ratings
from .onnx_policy import Lc0OnnxPolicyModel
from .stockfish_evaluation import _load_model_from_logdir
from .tournament_game_worker import play_tournament_worker
from .utils import (
    autocast,
    get_model_embeddings,
    load_run_config,
    padded_legal_logits,
    resolve_checkpoint,
    resolve_path,
    stack_positions,
    load_openings
)


ROUTER_CLASSES = {
    'activity': ('strategical', 'tactical'),
    'evaluation': ('winning', 'drawing', 'losing'),
}
CONSORTIUM_GROUPS = {
    'phase': ('opening', 'middlegame', 'endgame'),
    **ROUTER_CLASSES,
}


def _prepare_models(configured_models, cli_logdirs, root_dir: Path):
    specs = list(configured_models or [])
    specs.extend({'logdir': path} for path in cli_logdirs)
    if len(specs) < 2:
        raise ValueError('An all-against-all tournament needs at least two models')

    models = {}
    metadata = {}
    encodings = {}
    device = None
    for index, spec in enumerate(specs, start=1):
        if isinstance(spec, str):
            spec = {'logdir': spec}
        if isinstance(spec, dict) and 'onnx' in spec:
            checkpoint = resolve_path(spec['onnx'], root_dir)
            name = str(spec.get('name') or checkpoint.stem)
            if name in models:
                raise ValueError(f'Duplicate tournament model name {name!r}')
            model = Lc0OnnxPolicyModel(
                checkpoint, input_format=int(spec['input_format']),
                policy_output=spec.get('policy_output'),
                runtime=spec.get('runtime', 'onnxruntime'),
            )
            models[name] = model
            encodings[name] = f'lc0_native:{model.input_format}'
            metadata[name] = {
                'kind': 'lc0_onnx', 'name': name, 'logdir': str(checkpoint.parent),
                'checkpoint': str(checkpoint), 'encoding': encodings[name],
                'providers': model.providers,
            }
            print(f'Loaded {index}/{len(specs)}: {name} ({checkpoint})')
            continue
        if not isinstance(spec, dict) or 'logdir' not in spec:
            raise ValueError(f'Each model must be a logdir string or mapping with logdir: {spec!r}')

        logdir = resolve_path(spec['logdir'], root_dir)
        if not logdir.is_dir():
            raise NotADirectoryError(f'Model logdir not found: {logdir}')
        explicit_name = spec.get('name')
        name = str(explicit_name or logdir.name)
        if name in models:
            if explicit_name:
                raise ValueError(f'Duplicate tournament model name {name!r}; set distinct model names')
            name = f'{logdir.parent.name}/{logdir.name}'
            suffix = 2
            base_name = name
            while name in models:
                name = f'{base_name} ({suffix})'
                suffix += 1

        explicit_checkpoint = spec.get('checkpoint')
        checkpoint = None
        if explicit_checkpoint is not None:
            checkpoint_path = resolve_path(explicit_checkpoint, root_dir)
            checkpoint = resolve_checkpoint(logdir, '', str(checkpoint_path))
        model, encoding, run_config, checkpoint, model_device = _load_model_from_logdir(
            logdir,
            str(checkpoint) if checkpoint is not None else None,
        )
        if device is None:
            device = model_device
        elif model_device != device:
            model.to(device)
            model.eval()

        models[name] = model
        encodings[name] = encoding
        metadata[name] = {
            'kind': 'model',
            'name': name,
            'logdir': str(logdir),
            'checkpoint': str(checkpoint),
            'encoding': encoding,
            'training_name': run_config.get('training', {}).get('name'),
            'type': run_config.get('type'),
        }
        print(f'Loaded {index}/{len(specs)}: {name} ({checkpoint})')

    return models, encodings, metadata, device or torch.device('cpu')


def _load_router_for_tournament(
    router_spec: dict,
    route_groups: set[str],
    models: dict,
    encodings: dict[str, str],
    participant_metadata: dict,
    root_dir: Path,
    device,
):
    if 'classification_logdir' not in router_spec:
        raise ValueError('consortium.router.classification_logdir is required')
    selected_base = router_spec.get('base_model')
    separate_base = router_spec.get('base_model_logdir')
    if bool(selected_base) == bool(separate_base):
        raise ValueError(
            'Set exactly one of consortium.router.base_model (a selected model name) '
            'or base_model_logdir'
        )

    classification_logdir = resolve_path(router_spec['classification_logdir'], root_dir)
    classification_config = load_run_config(classification_logdir)
    if classification_config.get('type') != 'classification':
        raise ValueError('consortium.router.classification_logdir must be a classification run')
    classification_training = classification_config.get('training', {})
    classification_checkpoint = router_spec.get('classification_checkpoint')
    if classification_checkpoint is not None:
        classification_checkpoint = str(resolve_path(classification_checkpoint, root_dir))
    classification_checkpoint = resolve_checkpoint(
        classification_logdir,
        classification_training['name'],
        classification_checkpoint,
    )
    router_encoding = classification_config['data']['encoding']
    model_config = build_model_config(classification_config['model'])

    if selected_base:
        if selected_base not in models:
            raise ValueError(f'consortium.router.base_model {selected_base!r} is not a selected model')
        if participant_metadata[selected_base].get('type') != 'ssl':
            raise ValueError('consortium.router.base_model must refer to a selected SSL model')
        base_encoding = encodings[selected_base]
        base_info = {
            'base_model': selected_base,
            'base_model_logdir': participant_metadata[selected_base]['logdir'],
            'base_model_checkpoint': participant_metadata[selected_base]['checkpoint'],
        }
        base_hidden_size = getattr(getattr(models[selected_base], 'config', None), 'hidden_size', None)
        if base_hidden_size is not None and base_hidden_size != model_config.hidden_size:
            raise ValueError('Selected router base and classification heads have different hidden sizes')
    else:
        base_logdir = resolve_path(separate_base, root_dir)
        base_config = load_run_config(base_logdir)
        if base_config.get('type') != 'ssl':
            raise ValueError('consortium.router.base_model_logdir must be an SSL run')
        base_training = base_config.get('training', {})
        base_checkpoint = router_spec.get('base_model_checkpoint')
        if base_checkpoint is not None:
            base_checkpoint = str(resolve_path(base_checkpoint, root_dir))
        base_checkpoint = resolve_checkpoint(base_logdir, base_training['name'], base_checkpoint)
        base_encoding = base_config['data']['encoding']
        base_info = {
            'base_model_logdir': str(base_logdir),
            'base_model_checkpoint': str(base_checkpoint),
        }

    if base_encoding != router_encoding:
        raise ValueError('Router classifier and base model must use the same board encoding')

    if selected_base:
        router_base = models[selected_base]
    else:
        router_base = SSLChessModel(SSLModelConfig(**vars(model_config)))
        load_model_state_dict(
            router_base,
            torch.load(base_checkpoint, map_location='cpu', weights_only=True),
        )
        router_base.to(device).eval()
    router_heads = PositionClassificationHeads(model_config.hidden_size)
    router_heads.load_state_dict(
        torch.load(classification_checkpoint, map_location='cpu', weights_only=True)
    )
    router_heads.to(device).eval()

    return {
        'base': router_base,
        'heads': router_heads,
        'encoding': router_encoding,
        'groups': tuple(sorted(route_groups)),
    }, {
        'classification_logdir': str(classification_logdir),
        'classification_checkpoint': str(classification_checkpoint),
        **base_info,
        'encoding': router_encoding,
    }


def _load_tournament_consortium(
    spec: dict,
    models: dict,
    encodings: dict[str, str],
    shared_router: dict | None,
    shared_router_metadata: dict | None,
):
    if not isinstance(spec, dict):
        raise ValueError('Each consortium entry must be a mapping')

    name = str(spec.get('name') or 'consortium')
    groups = spec.get('expert_groups', [])
    supported = set(CONSORTIUM_GROUPS)
    if not groups or len(groups) != len(set(groups)) or not set(groups) <= supported:
        raise ValueError(f'consortium.expert_groups must be a non-empty unique subset of {sorted(supported)}')

    selected_experts = {}
    for group in groups:
        group_spec = spec.get('experts', {}).get(group, {})
        missing = set(CONSORTIUM_GROUPS[group]) - set(group_spec)
        extra = set(group_spec) - set(CONSORTIUM_GROUPS[group])
        if missing or extra:
            raise ValueError(
                f'consortium.experts.{group} must map exactly {list(CONSORTIUM_GROUPS[group])}; '
                f'missing={sorted(missing)}, extra={sorted(extra)}'
            )
        selected_experts[group] = {}
        for label in CONSORTIUM_GROUPS[group]:
            model_name = group_spec[label]
            if not isinstance(model_name, str) or model_name not in models:
                raise ValueError(
                    f'consortium expert {group}/{label} must name a selected model; '
                    f'got {model_name!r}'
                )
            selected_experts[group][label] = model_name

    route_groups = set(groups) & set(ROUTER_CLASSES)
    if route_groups and shared_router is None:
        raise ValueError(
            f'Consortium {spec.get("name", "consortium")!r} uses activity/evaluation '
            'but no shared top-level router is configured'
        )
    router = shared_router if route_groups else None
    router_metadata = shared_router_metadata if route_groups else None

    routed_encodings = set()
    for group in groups:
        routed_encodings.update(encodings[model_name] for model_name in selected_experts[group].values())
    if router is not None:
        routed_encodings.add(router['encoding'])

    metadata = {
        'kind': 'consortium',
        'name': name,
        'expert_groups': list(groups),
        'experts': selected_experts,
        'router': router_metadata,
    }
    consortium = {
        'name': name,
        'groups': tuple(groups),
        'experts': selected_experts,
        'router': router,
    }
    return consortium, tuple(sorted(routed_encodings)), metadata


def _resolve_shared_router_spec(config_router, specs):
    nested_routers = [
        spec['router']
        for spec in specs
        if isinstance(spec, dict) and spec.get('router') is not None
    ]
    if config_router is not None:
        if any(router != config_router for router in nested_routers):
            raise ValueError('Consortiums must all use the single top-level router configuration')
        return config_router
    if nested_routers:
        shared = nested_routers[0]
        if any(router != shared for router in nested_routers[1:]):
            raise ValueError('Only one shared router is supported; consortium router settings differ')
        return shared
    return None


def _load_tournament_consortiums(
    specs,
    models,
    encodings,
    participant_metadata,
    root_dir,
    device,
    router_spec,
):
    if specs is None:
        specs = []
    if not isinstance(specs, list):
        raise ValueError('consortiums must be a list of consortium mappings')

    route_groups = set()
    for spec in specs:
        if not isinstance(spec, dict):
            raise ValueError('Each consortium entry must be a mapping')
        route_groups.update(set(spec.get('expert_groups', [])) & set(ROUTER_CLASSES))
    shared_router = None
    shared_router_metadata = None
    if route_groups:
        if router_spec is None:
            raise ValueError('Activity/evaluation consortiums require one shared top-level router')
        if not isinstance(router_spec, dict):
            raise ValueError('router must be a mapping')
        shared_router, shared_router_metadata = _load_router_for_tournament(
            router_spec,
            route_groups,
            models,
            encodings,
            participant_metadata,
            root_dir,
            device,
        )

    consortiums = {}
    metadata = {}
    routed_encodings = set()
    for spec in specs:
        consortium, encoding_list, item_metadata = _load_tournament_consortium(
            spec,
            models,
            encodings,
            shared_router,
            shared_router_metadata,
        )
        name = consortium['name']
        if name in models or name in consortiums:
            raise ValueError(f'Duplicate consortium/participant name {name!r}')
        consortiums[name] = consortium
        metadata[name] = item_metadata
        routed_encodings.update(encoding_list)
    return consortiums, tuple(sorted(routed_encodings)), metadata


def _validate_config(config: TournamentConfig):
    if config.rounds <= 0:
        raise ValueError('rounds must be positive')
    if config.play_mode not in ('value', 'policy', 'sampled'):
        raise ValueError("play_mode must be 'value', 'policy', or 'sampled'")
    if (
        config.num_workers <= 0
        or config.max_active_matches <= 0
        or config.batch_size <= 0
        or config.value_batch_size <= 0
        or config.model_output_cache_size < 0
    ):
        raise ValueError('Worker/concurrency and batch settings must be positive; cache size cannot be negative')
    if config.batch_timeout_ms < 0:
        raise ValueError('batch_timeout_ms cannot be negative')
    if config.max_game_plies <= 0:
        raise ValueError('max_game_plies must be positive')
    if config.prior_elo_std <= 0 or config.elo_min >= config.elo_max:
        raise ValueError('Invalid Bayesian Elo prior or rating bounds')
    if config.pin_player is not None:
        if not isinstance(config.pin_player, dict) or set(config.pin_player) != {'name', 'elo'}:
            raise ValueError('pin_player must be a mapping with exactly `name` and `elo`')
        if not isinstance(config.pin_player['name'], str) or not config.pin_player['name']:
            raise ValueError('pin_player.name must be a non-empty participant name')
        pinned_elo = config.pin_player['elo']
        if (
            isinstance(pinned_elo, bool)
            or not isinstance(pinned_elo, (int, float))
            or not np.isfinite(pinned_elo)
            or not config.elo_min <= pinned_elo <= config.elo_max
        ):
            raise ValueError(
                f'pin_player.elo must be finite and within '
                f'[{config.elo_min}, {config.elo_max}]'
            )


def _standings_order(names, match_points, game_score):
    return sorted(
        names,
        key=lambda name: (
            -match_points[name],
            -game_score[name],
            name.casefold(),
        ),
    )


def _unique_positions(positions, stats=None):
    """Deduplicate equal encoded boards so identical games share one forward pass."""
    unique = []
    unique_indices = {}
    mapping = []
    for index, position in enumerate(positions):
        key = position.get('position_key')
        if key is None:
            key = ('unique', index)
        unique_index = unique_indices.get(key)
        if unique_index is None:
            unique_index = len(unique)
            unique_indices[key] = unique_index
            unique.append(position)
        mapping.append(unique_index)
    if stats is not None:
        stats['unique_input_positions'] += len(unique)
        stats['duplicate_positions_skipped'] += len(positions) - len(unique)
    return unique, mapping


def _model_outputs(model, positions, device, *, policy: bool = False, value: bool = False):
    if isinstance(model, Lc0OnnxPolicyModel):
        if value:
            raise ValueError('LC0 ONNX tournament participants require policy or sampled play')
        return torch.from_numpy(model.predict_positions(positions)).to(device), None
    tokens, extra = stack_positions(positions, device)
    with torch.inference_mode(), autocast(device):
        squares, cls = get_model_embeddings(model, tokens, extra)
        policy_logits = model.policy_head(squares, None).logits if policy else None
        values = None
        if value:
            # SSL and PV value outputs regress the side-to-move expected result.
            values = model.value_head(cls, None).logits.float().mean(dim=-1)
    return policy_logits, values


def _cached_model_outputs(
    model,
    model_name,
    positions,
    device,
    output_kind,
    inference_cache,
    cache_limit,
    batch_size,
    stats,
):
    """Run only cache misses and retain a bounded set of deterministic model outputs."""

    if output_kind not in ('policy', 'value'):
        raise ValueError(f'Unsupported cached model output: {output_kind}')
    model_cache = inference_cache.setdefault(model_name, {}).setdefault(
        output_kind, OrderedDict()
    )
    values = [None] * len(positions)
    misses = []
    miss_rows = []
    miss_keys = []

    for row, position in enumerate(positions):
        key = position.get('position_key')
        if key is not None and cache_limit > 0 and key in model_cache:
            model_cache.move_to_end(key)
            values[row] = model_cache[key]
            stats['model_output_cache_hits'] += 1
        else:
            misses.append(position)
            miss_rows.append(row)
            miss_keys.append(key)
            stats['model_output_cache_misses'] += 1

    for offset in range(0, len(misses), batch_size):
        chunk = misses[offset:offset + batch_size]
        if output_kind == 'policy':
            logits, _ = _model_outputs(model, chunk, device, policy=True)
            chunk_values = logits.detach().float().cpu().numpy()
        else:
            _, predictions = _model_outputs(model, chunk, device, value=True)
            chunk_values = predictions.detach().float().cpu().tolist()

        for local_index, output in enumerate(chunk_values):
            row = miss_rows[offset + local_index]
            key = miss_keys[offset + local_index]
            value = np.asarray(output, dtype=np.float32) if output_kind == 'policy' else float(output)
            values[row] = value
            if key is not None and cache_limit > 0:
                model_cache[key] = value
                model_cache.move_to_end(key)
                while len(model_cache) > cache_limit:
                    model_cache.popitem(last=False)

    if output_kind == 'policy':
        return torch.from_numpy(np.stack(values)).to(device)
    return torch.as_tensor(values, dtype=torch.float32, device=device)


def _base_model_worker(base_model, positions, device):
    """Encode positions once with the router base model; return its square embeddings."""
    tokens, extra = stack_positions(positions, device)
    with torch.inference_mode(), autocast(device):
        squares, _ = get_model_embeddings(base_model, tokens, extra)
    return squares


def _router_worker(router_heads, square_embeddings, device):
    """Run only the activity/evaluation classification heads on base embeddings."""
    with torch.inference_mode(), autocast(device):
        logits, _ = router_heads(square_embeddings)
    return {
        group: torch.softmax(group_logits.float(), dim=-1).cpu().numpy()
        for group, group_logits in logits.items()
    }


def _batch_legal_probabilities(logits, legal_ids):
    padded, lengths, legal_logits = padded_legal_logits(logits, legal_ids)
    probabilities = torch.softmax(legal_logits, dim=-1).cpu().numpy()
    return padded, lengths, probabilities


def _batch_choose_moves(logits, requests, rng):
    padded, lengths, legal_logits = padded_legal_logits(
        logits, [request['legal_ids'] for request in requests]
    )
    if requests[0]['play_mode'] == 'policy':
        choices = legal_logits.argmax(dim=1).cpu().numpy()
    else:
        probabilities = torch.softmax(legal_logits, dim=-1).cpu().numpy()
        choices = np.asarray([
            rng.choice(length, p=probabilities[row, :length])
            for row, length in enumerate(lengths)
        ], dtype=np.int64)
    return [int(padded[row, choice]) for row, choice in enumerate(choices)]


def _route_weights_for_items(
    consortia, route_items, device, batch_size, inference_cache, cache_limit, stats
):
    """Route positions for several consortiums through the one shared router batch."""
    results = [None] * len(route_items)
    for offset in range(0, len(route_items), batch_size):
        chunk = route_items[offset:offset + batch_size]
        router_rows = [
            index for index, (name, _) in enumerate(chunk)
            if consortia[name]['router'] is not None
        ]
        classifier_probs_by_row = {}
        if router_rows:
            router = consortia[chunk[router_rows[0]][0]]['router']
            if any(consortia[chunk[index][0]]['router'] is not router for index in router_rows):
                raise RuntimeError('Consortiums must share the same router instance')
            router_cache = inference_cache.setdefault('__shared_router__', {}).setdefault(
                'classification', OrderedDict()
            )
            encoded = [
                chunk[index][1]['routed_positions'][router['encoding']]
                for index in router_rows
            ]
            miss_encoded = []
            miss_rows = []
            miss_keys = []
            for row, position in zip(router_rows, encoded):
                key = position.get('position_key')
                if key is not None and cache_limit > 0 and key in router_cache:
                    router_cache.move_to_end(key)
                    classifier_probs_by_row[row] = router_cache[key]
                    stats['router_cache_hits'] += 1
                else:
                    miss_encoded.append(position)
                    miss_rows.append(row)
                    miss_keys.append(key)
                    stats['router_cache_misses'] += 1

            if miss_encoded:
                unique_encoded, miss_map = _unique_positions(miss_encoded, stats)
                base_embeddings = _base_model_worker(router['base'], unique_encoded, device)
                unique_probs = _router_worker(router['heads'], base_embeddings, device)
                for miss_index, (row, key) in enumerate(zip(miss_rows, miss_keys)):
                    output = {
                        group: probabilities[miss_map[miss_index]]
                        for group, probabilities in unique_probs.items()
                    }
                    classifier_probs_by_row[row] = output
                    if key is not None and cache_limit > 0:
                        router_cache[key] = output
                        router_cache.move_to_end(key)
                        while len(router_cache) > cache_limit:
                            router_cache.popitem(last=False)

        for row, (name, position) in enumerate(chunk):
            consortium = consortia[name]
            groups = consortium['groups']
            experts = consortium['experts']
            group_scale = 1.0 / len(groups)
            model_weights = {}
            for group in groups:
                if group == 'phase':
                    label = position['phase']
                    if label not in experts[group]:
                        raise ValueError(f'Unknown phase label {label!r} for consortium routing')
                    model_name = experts[group][label]
                    model_weights[model_name] = model_weights.get(model_name, 0.0) + group_scale
                else:
                    probabilities = classifier_probs_by_row[row][group]
                    for class_index, label in enumerate(ROUTER_CLASSES[group]):
                        model_name = experts[group][label]
                        model_weights[model_name] = (
                            model_weights.get(model_name, 0.0)
                            + group_scale * float(probabilities[class_index])
                        )
            results[offset + row] = model_weights
    return results


def _infer_consortium_policy_batch(
    models, encodings, consortia, device, items, rng, batch_size,
    inference_cache, cache_limit, stats,
):
    route_items = [
        (name, {'phase': request['phase'], 'routed_positions': request['routed_positions']})
        for name, request in items
    ]
    weights = _route_weights_for_items(
        consortia, route_items, device, batch_size, inference_cache, cache_limit, stats
    )
    votes = [np.zeros(len(request['legal_ids']), dtype=np.float64) for _, request in items]
    tasks_by_model = {}
    for item_index, model_weights in enumerate(weights):
        for model_name, weight in model_weights.items():
            if weight > 0.0:
                tasks_by_model.setdefault(model_name, []).append((item_index, weight))

    for model_name, tasks in tasks_by_model.items():
        positions = [
            items[item_index][1]['routed_positions'][encodings[model_name]]
            for item_index, _ in tasks
        ]
        unique_positions, task_rows = _unique_positions(positions, stats)
        logits = _cached_model_outputs(
            models[model_name], model_name, unique_positions, device, 'policy',
            inference_cache, cache_limit, batch_size, stats,
        )
        task_logits = logits[task_rows]
        task_legal_ids = [items[item_index][1]['legal_ids'] for item_index, _ in tasks]
        _, legal_lengths, probabilities = _batch_legal_probabilities(task_logits, task_legal_ids)
        for row, (item_index, weight) in enumerate(tasks):
            votes[item_index] += probabilities[row, :legal_lengths[row]] * weight

    move_ids = []
    for (_, request), policy in zip(items, votes):
        if request['play_mode'] == 'policy':
            choice = int(np.argmax(policy))
        else:
            distribution = policy / policy.sum()
            choice = int(rng.choice(len(distribution), p=distribution))
        move_ids.append(int(request['legal_ids'][choice]))
    return move_ids


def _infer_consortium_value_batch(
    models, encodings, consortia, device, items, batch_size,
    inference_cache, cache_limit, stats,
):
    route_items = []
    child_map = []
    values = [list(request['terminal_values']) for _, request in items]
    for item_index, (name, request) in enumerate(items):
        for child_index, routed_positions in enumerate(request['routed_child_positions']):
            if routed_positions is not None:
                route_items.append((name, {
                    'phase': request['child_phases'][child_index],
                    'routed_positions': routed_positions,
                }))
                child_map.append((item_index, child_index))

    if route_items:
        weights = _route_weights_for_items(
            consortia, route_items, device, batch_size, inference_cache, cache_limit, stats
        )
        value_sums = [0.0] * len(route_items)
        tasks_by_model = {}
        for position_index, model_weights in enumerate(weights):
            for model_name, weight in model_weights.items():
                if weight > 0.0:
                    tasks_by_model.setdefault(model_name, []).append((position_index, weight))

        for model_name, tasks in tasks_by_model.items():
            positions = [
                route_items[position_index][1]['routed_positions'][encodings[model_name]]
                for position_index, _ in tasks
            ]
            unique_positions, task_rows = _unique_positions(positions, stats)
            unique_predictions = _cached_model_outputs(
                models[model_name], model_name, unique_positions, device, 'value',
                inference_cache, cache_limit, batch_size, stats,
            ).cpu().tolist()
            for task_index, (position_index, weight) in enumerate(tasks):
                value_sums[position_index] += float(unique_predictions[task_rows[task_index]]) * weight
        for (item_index, child_index), value in zip(child_map, value_sums):
            values[item_index][child_index] = value

    move_ids = []
    for (_, request), child_values in zip(items, values):
        child_values = np.asarray(child_values, dtype=np.float64)
        if not np.all(np.isfinite(child_values)):
            raise ValueError('Consortium value routing produced a missing child value')
        move_ids.append(int(request['legal_ids'][int(np.argmin(child_values))]))
    return move_ids


def _infer_batch(
    models, encodings, consortia, device, batch, rng, config,
    inference_cache, stats,
):
    stats['request_batches'] += 1
    stats['requests'] += len(batch)
    stats['max_request_batch'] = max(stats['max_request_batch'], len(batch))
    grouped = {}
    for request in batch:
        grouped.setdefault(request['model_id'], []).append(request)

    consortium_items = []
    for model_id, requests in grouped.items():
        stats['model_batches'] += 1
        stats['model_batch_requests'] += len(requests)
        stats['max_model_batch'] = max(stats['max_model_batch'], len(requests))
        model_stat = stats['per_model'].setdefault(
            model_id, {'batches': 0, 'requests': 0, 'max_batch': 0}
        )
        model_stat['batches'] += 1
        model_stat['requests'] += len(requests)
        model_stat['max_batch'] = max(model_stat['max_batch'], len(requests))

        if model_id in consortia:
            consortium_items.extend((model_id, request) for request in requests)
            continue

        try:
            model = models[model_id]
            if requests[0]['play_mode'] == 'value':
                child_positions = []
                child_indices = []
                scores = [list(request['terminal_values']) for request in requests]
                for request_index, request in enumerate(requests):
                    for child_index, position in enumerate(request['child_positions']):
                        if position is not None:
                            child_positions.append(position)
                            child_indices.append((request_index, child_index))
                if child_positions:
                    unique_children, child_rows = _unique_positions(child_positions, stats)
                    unique_values = _cached_model_outputs(
                        model, model_id, unique_children, device, 'value',
                        inference_cache, config.model_output_cache_size,
                        config.value_batch_size, stats,
                    ).cpu().tolist()
                    for (request_index, child_index), unique_index in zip(child_indices, child_rows):
                        scores[request_index][child_index] = float(unique_values[unique_index])
                move_ids = [
                    int(request['legal_ids'][int(np.argmin(np.asarray(child_scores)))])
                    for request, child_scores in zip(requests, scores)
                ]
            else:
                unique_requests, request_rows = _unique_positions(requests, stats)
                logits = _cached_model_outputs(
                    model, model_id, unique_requests, device, 'policy',
                    inference_cache, config.model_output_cache_size,
                    config.batch_size, stats,
                ).float()
                move_ids = _batch_choose_moves(logits[request_rows], requests, rng)
            for request, move_id in zip(requests, move_ids):
                request['response_queue'].put({'move_id': move_id})
        except BaseException as exc:
            message = f'{type(exc).__name__}: {exc}'
            for request in requests:
                request['response_queue'].put({'error': message})

    if consortium_items:
        try:
            if consortium_items[0][1]['play_mode'] == 'value':
                move_ids = _infer_consortium_value_batch(
                    models,
                    encodings,
                    consortia,
                    device,
                    consortium_items,
                    config.value_batch_size,
                    inference_cache,
                    config.model_output_cache_size,
                    stats,
                )
            else:
                move_ids = _infer_consortium_policy_batch(
                    models,
                    encodings,
                    consortia,
                    device,
                    consortium_items,
                    rng,
                    config.batch_size,
                    inference_cache,
                    config.model_output_cache_size,
                    stats,
                )
            for (_, request), move_id in zip(consortium_items, move_ids):
                request['response_queue'].put({'move_id': move_id})
        except BaseException as exc:
            message = f'{type(exc).__name__}: {exc}'
            for _, request in consortium_items:
                request['response_queue'].put({'error': message})


def _run_inference_worker(
    models,
    encodings,
    consortia,
    device,
    config,
    request_queue,
    response_queues,
    processes,
    result_queue,
    match_ids,
    on_match_result,
    rng,
    inference_cache,
    stats,
):
    results = {}
    dead_since = {}
    expected_ids = set(match_ids)

    service_started = time.monotonic()
    first_request_received = False
    while len(results) < len(match_ids):
        batch = []
        wait_started = time.monotonic()
        try:
            batch.append(request_queue.get(timeout=0.01))
        except queue.Empty:
            pass
        stats['request_queue_wait_seconds'] += time.monotonic() - wait_started
        if batch and not first_request_received:
            stats['first_request_latency_seconds'] = time.monotonic() - service_started
            first_request_received = True

        if batch:
            collection_started = time.monotonic()
            deadline = collection_started + config.batch_timeout_ms / 1000.0
            while len(batch) < config.batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                wait_started = time.monotonic()
                try:
                    request = request_queue.get(timeout=remaining)
                except queue.Empty:
                    stats['batch_collection_wait_seconds'] += time.monotonic() - wait_started
                    break
                stats['batch_collection_wait_seconds'] += time.monotonic() - wait_started
                request['response_queue'] = response_queues[request['game_id']]
                batch.append(request)
            stats['batch_collection_seconds'] += time.monotonic() - collection_started
            batch[0]['response_queue'] = response_queues[batch[0]['game_id']]
            inference_started = time.monotonic()
            _infer_batch(
                models, encodings, consortia, device, batch, rng, config,
                inference_cache, stats,
            )
            stats['inference_seconds'] += time.monotonic() - inference_started

        while True:
            try:
                event = result_queue.get_nowait()
            except queue.Empty:
                break
            match_id = event.get('match_id')
            if match_id in expected_ids and match_id not in results:
                results[match_id] = event
                on_match_result(event)

        for match_id, process in processes.items():
            if match_id in results or process.exitcode is None:
                continue
            dead_at = dead_since.setdefault(match_id, time.monotonic())
            if time.monotonic() - dead_at > 1.0:
                event = {
                    'match_id': match_id,
                    'error': f'Game worker exited with code {process.exitcode} without a result',
                }
                results[match_id] = event
                on_match_result(event)

    stats['service_seconds'] += time.monotonic() - service_started
    return [results[match_id] for match_id in match_ids]


def _new_inference_stats():
    return {
        'requests': 0,
        'request_batches': 0,
        'max_request_batch': 0,
        'model_batches': 0,
        'model_batch_requests': 0,
        'max_model_batch': 0,
        'unique_input_positions': 0,
        'duplicate_positions_skipped': 0,
        'model_output_cache_hits': 0,
        'model_output_cache_misses': 0,
        'router_cache_hits': 0,
        'router_cache_misses': 0,
        'per_model': {},
        'first_request_latency_seconds': 0.0,
        'request_queue_wait_seconds': 0.0,
        'batch_collection_wait_seconds': 0.0,
        'batch_collection_seconds': 0.0,
        'inference_seconds': 0.0,
        'service_seconds': 0.0,
    }


def _format_inference_stats(stats):
    mean_batch = stats['requests'] / max(stats['request_batches'], 1)
    mean_model_batch = stats['model_batch_requests'] / max(stats['model_batches'], 1)
    queue_wait = stats['request_queue_wait_seconds']
    collection_wait = stats['batch_collection_wait_seconds']
    inference = stats['inference_seconds']
    service = stats['service_seconds']
    return (
        f"Inference: {stats['requests']} positions; request batches "
        f"{mean_batch:.1f} avg/{stats['max_request_batch']} max; per-model batches "
        f"{mean_model_batch:.1f} avg/{stats['max_model_batch']} max; "
        f"unique input rows {stats['unique_input_positions']} "
        f"(skipped {stats['duplicate_positions_skipped']} duplicates); "
        f"output-cache {stats['model_output_cache_hits']} hits/"
        f"{stats['model_output_cache_misses']} misses; router-cache "
        f"{stats['router_cache_hits']} hits/{stats['router_cache_misses']} misses; "
        f"first request {stats['first_request_latency_seconds']:.1f}s, "
        f"queue idle {queue_wait:.1f}s, batch wait {collection_wait:.1f}s, "
        f"inference {inference:.1f}s / service {service:.1f}s"
    )


def _game_score_for_model(game, model_name):
    if game['white_model'] == model_name:
        return float(game['white_score'])
    return 1.0 - float(game['white_score'])


def _match_result(event):
    games = event.get('games', [])
    if len(games) != 2:
        return None
    model_a_score = sum(_game_score_for_model(game, event['model_a']) for game in games)
    model_b_score = 2.0 - model_a_score
    points_a = 1.0 if model_a_score > model_b_score else 0.0 if model_a_score < model_b_score else 0.5
    return {
        'num_games': 2,
        'model_a_score': float(model_a_score),
        'model_b_score': float(model_b_score),
        'model_a_match_points': points_a,
        'model_b_match_points': 1.0 - points_a,
        'result': 'win' if points_a == 1.0 else 'loss' if points_a == 0.0 else 'draw',
    }


def _leaderboard_preview(names, match_points, game_score):
    top = _standings_order(names, match_points, game_score)[:3]
    return ' | '.join(
        f'{rank}. {name} {match_points[name]:g}pt'
        for rank, name in enumerate(top, 1)
    )


def _write_markdown(path: Path, output: dict):
    def cell(value):
        return str(value).replace('|', '&#124;').replace('\n', ' ')

    lines = [
        '| Rank | Participant | Elo | Match points | Game score | Games | W-D-L |',
        '|---:|---|---:|---:|---:|---:|---|',
    ]
    names = [row['name'] for row in output['rankings']]
    for row in output['rankings']:
        lines.append(
            f"| {row['rank']} | {cell(row['name'])} | "
            f"{row['elo']:.0f} ± {row['elo_std']:.0f} | "
            f"{row['match_points']:g} | {row['game_score']:g} | {row['games']} | "
            f"{row['wins']}-{row['draws']}-{row['losses']} |"
        )
    head_to_head = {(a, b): [0, 0, 0] for a in names for b in names if a != b}
    for record in output['rounds']:
        for match in record['matches']:
            for game in match.get('games', []):
                for name, opponent in ((game['white_model'], game['black_model']),
                                       (game['black_model'], game['white_model'])):
                    score = _game_score_for_model(game, name)
                    head_to_head[name, opponent][0 if score == 1 else 1 if score == .5 else 2] += 1
    lines.extend(['', '| Head-to-head (row W-D-L) | ' + ' | '.join(map(cell, names)) + ' |',
                  '|---|' + '---:|' * len(names)])
    for name in names:
        cells = ['—' if name == opponent else '-'.join(map(str, head_to_head[name, opponent]))
                 for opponent in names]
        lines.append('| ' + cell(name) + ' | ' + ' | '.join(cells) + ' |')
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def _append_match_pgn(path: Path, event: dict):
    """Append completed games immediately so a long run retains its games."""
    with path.open('a', encoding='utf-8') as handle:
        for index, result in enumerate(event.get('games', []), 1):
            game = chess.pgn.Game()
            game.setup(chess.Board(event['opening_fen'], chess960=True))
            game.headers.update({
                'Event': 'Bumblebot all-against-all tournament',
                'Round': f"{event['round']}.{event['match_id'] + 1}.{index}",
                'White': result['white_model'], 'Black': result['black_model'],
                'Result': result['result'],
                'Termination': 'unterminated' if result['termination'] == 'max_game_plies' else 'normal',
                'TerminationReason': result['termination'],
                'PlyCount': str(result['plies']),
            })
            board = game.board()
            node = game
            for san in result['moves'].split():
                move = board.parse_san(san)
                node = node.add_variation(move)
                board.push(move)
            print(game.accept(chess.pgn.StringExporter(headers=True, variations=False, comments=False)),
                  file=handle, end='\n\n')


def run_tournament(
    tournament_config_path: str,
    *,
    logdirs: list[str] | None = None,
    book_path: str | None = None,
) -> dict:
    config_path = Path(tournament_config_path).expanduser().resolve()
    raw_config = load_config_file(str(config_path))
    config = TournamentConfig(**raw_config)
    _validate_config(config)
    root_dir = Path.cwd()
    cli_logdirs = [str(resolve_path(path, root_dir)) for path in (logdirs or [])]
    models, encodings, participants, device = _prepare_models(
        config.models, cli_logdirs, root_dir
    )

    if config.consortium is not None:
        if config.consortiums:
            raise ValueError('Use consortiums or the legacy consortium field, not both')
        consortium_specs = [config.consortium]
    else:
        consortium_specs = config.consortiums or []
    router_spec = _resolve_shared_router_spec(config.router, consortium_specs)
    consortia, routed_encodings, consortium_metadata = _load_tournament_consortiums(
        consortium_specs,
        models,
        encodings,
        participants,
        root_dir,
        device,
        router_spec,
    )
    participants.update(consortium_metadata)
    consortium_names = tuple(consortia)
    entrant_names = [*models, *consortium_names]
    if config.pin_player is not None and config.pin_player['name'] not in entrant_names:
        raise ValueError(
            f'pin_player.name {config.pin_player["name"]!r} is not a tournament participant; '
            f'available participants: {entrant_names}'
        )

    if book_path is not None:
        resolved_book = resolve_path(book_path, root_dir)
    elif config.book_path is not None:
        resolved_book = resolve_path(config.book_path, root_dir)
    else:
        resolved_book = root_dir / 'book.csv'
    openings = load_openings(resolved_book)

    rng = random.Random(config.seed)
    inference_rng = np.random.default_rng(config.seed)
    inference_cache = {}
    context = mp.get_context('spawn')
    request_queue = context.Queue()
    result_queue = context.Queue()
    match_points = {name: 0.0 for name in entrant_names}
    game_score = {name: 0.0 for name in entrant_names}
    rounds = []
    all_games = []
    next_match_id = 0
    errors = []
    pairings = list(itertools.combinations(entrant_names, 2))

    result_path = resolve_path(config.results_path, root_dir) if config.results_path else root_dir / 'tournament_results.md'
    markdown_path = result_path.with_suffix('.md')
    pgn_path = result_path.with_suffix('.pgn')
    pgn_path.parent.mkdir(parents=True, exist_ok=True)
    pgn_path.write_text('', encoding='utf-8')

    print(
        f'Starting {config.rounds}-round all-against-all tournament: '
        f'{len(entrant_names)} participants, {len(pairings)} paired matchups per round, '
        f'{config.play_mode} play'
    )
    try:
        for round_number in range(1, config.rounds + 1):
            opening_fen = rng.choice(openings)
            round_record = {
                'round': round_number,
                'opening_fen': opening_fen,
                'matches': [],
            }
            rounds.append(round_record)
            match_specs = []
            for model_a, model_b in pairings:
                match_specs.append({
                    'match_id': next_match_id,
                    'round': round_number,
                    'model_a': model_a,
                    'model_b': model_b,
                    'opening_fen': opening_fen,
                })
                next_match_id += 1

            progress = tqdm(
                total=len(match_specs),
                desc=f'All-against-all round {round_number}/{config.rounds}',
            )
            inference_stats = _new_inference_stats()

            def on_match_result(event):
                if 'error' not in event:
                    computed = _match_result(event)
                    if computed is None:
                        event['error'] = 'A paired matchup must contain exactly two games'
                    else:
                        event.update(computed)
                        model_a = event['model_a']
                        model_b = event['model_b']
                        match_points[model_a] += computed['model_a_match_points']
                        match_points[model_b] += computed['model_b_match_points']
                        game_score[model_a] += sum(
                            _game_score_for_model(game, model_a) for game in event['games']
                        )
                        game_score[model_b] += sum(
                            _game_score_for_model(game, model_b) for game in event['games']
                        )
                        # Keep both color-swapped games as independent Elo observations.
                        all_games.extend(event['games'])
                        _append_match_pgn(pgn_path, event)
                if 'error' in event:
                    errors.append(event)
                round_record['matches'].append(event)
                progress.update(1)
                progress.set_postfix_str(
                    'Top 3: ' + _leaderboard_preview(entrant_names, match_points, game_score)
                )

            try:
                for offset in range(0, len(match_specs), config.max_active_matches):
                    wave = match_specs[offset:offset + config.max_active_matches]
                    response_queues = {
                        game_id: context.Queue()
                        for spec in wave
                        for game_id in (spec['match_id'] * 2, spec['match_id'] * 2 + 1)
                    }
                    worker_count = min(config.num_workers, len(wave))
                    worker_shards = [wave[index::worker_count] for index in range(worker_count)]
                    processes = {}
                    worker_processes = []
                    try:
                        for worker_index, shard in enumerate(worker_shards):
                            worker_response_queues = {
                                game_id: response_queues[game_id]
                                for spec in shard
                                for game_id in (spec['match_id'] * 2, spec['match_id'] * 2 + 1)
                            }
                            process = context.Process(
                                target=play_tournament_worker,
                                args=(
                                    shard,
                                    config.play_mode,
                                    config.max_game_plies,
                                    encodings,
                                    routed_encodings,
                                    consortium_names,
                                    request_queue,
                                    worker_response_queues,
                                    result_queue,
                                ),
                                name=f'tournament-game-worker-{worker_index}',
                            )
                            process.start()
                            worker_processes.append(process)
                            for spec in shard:
                                processes[spec['match_id']] = process

                        _run_inference_worker(
                            models,
                            encodings,
                            consortia,
                            device,
                            config,
                            request_queue,
                            response_queues,
                            processes,
                            result_queue,
                            [spec['match_id'] for spec in wave],
                            on_match_result,
                            inference_rng,
                            inference_cache,
                            inference_stats,
                        )
                    finally:
                        for process in worker_processes:
                            process.join(timeout=5.0)
                            if process.is_alive():
                                process.terminate()
                                process.join()
                        for channel in response_queues.values():
                            channel.close()
                            channel.join_thread()
            finally:
                progress.close()

            print(_format_inference_stats(inference_stats))
            round_record['inference'] = inference_stats
            round_record['matches'].sort(key=lambda match: match.get('match_id', -1))

    finally:
        for channel in (request_queue, result_queue):
            channel.close()
            channel.join_thread()

    elo_ratings = bayes_elo_ratings(
        entrant_names,
        all_games,
        prior_mean=config.prior_elo_mean,
        prior_std=config.prior_elo_std,
        elo_min=config.elo_min,
        elo_max=config.elo_max,
        pinned_ratings=(
            {config.pin_player['name']: config.pin_player['elo']}
            if config.pin_player is not None else None
        ),
    )
    game_stats = {
        name: {'wins': 0, 'draws': 0, 'losses': 0, 'games': 0}
        for name in entrant_names
    }
    for game in all_games:
        for name in (game['white_model'], game['black_model']):
            score = _game_score_for_model(game, name)
            game_stats[name]['games'] += 1
            if score == 1.0:
                game_stats[name]['wins'] += 1
            elif score == 0.5:
                game_stats[name]['draws'] += 1
            else:
                game_stats[name]['losses'] += 1

    # The table order is based on match points, then total game score, then Elo.
    order = sorted(
        entrant_names,
        key=lambda name: (
            -match_points[name],
            -game_score[name],
            -elo_ratings[name][0],
            name.casefold(),
        ),
    )
    rankings = []
    for rank, name in enumerate(order, start=1):
        elo, elo_std = elo_ratings[name]
        rankings.append({
            'rank': rank,
            'name': name,
            'elo': elo,
            'elo_std': elo_std,
            'match_points': match_points[name],
            'game_score': game_score[name],
            **game_stats[name],
        })

    output = {
        'summary': {
            'format': 'all_against_all_paired',
            'rounds': config.rounds,
            'play_mode': config.play_mode,
            'num_participants': len(entrant_names),
            'num_matchups_per_round': len(pairings),
            'num_matches': sum(len(record['matches']) for record in rounds),
            'num_games': len(all_games),
            'pin_player': config.pin_player,
            'seed': config.seed,
            'book': str(resolved_book),
            'device': str(device),
        },
        'participants': participants,
        'rankings': rankings,
        'rounds': rounds,
    }

    _write_markdown(markdown_path, output)

    print(f'Tournament Elo rankings ({config.play_mode}, paired all-against-all):')
    for ranking in rankings:
        print(
            f"{ranking['rank']:>2}. {ranking['name']}: "
            f"{ranking['elo']:.0f} ± {ranking['elo_std']:.0f} Elo, "
            f"{ranking['match_points']:g} match points"
        )
    print(f'Results: {markdown_path} and {pgn_path}')
    if errors:
        raise RuntimeError(
            f'{len(errors)} paired matchup(s) failed; partial results saved to {markdown_path} and {pgn_path}. '
            f"First failure: {errors[0].get('error', 'unknown error')}"
        )
    return output
