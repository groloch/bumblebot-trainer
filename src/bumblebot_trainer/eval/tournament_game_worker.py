"""CPU workers that play color-swapped model-vs-model game pairs."""

import hashlib
import threading
import traceback
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed

import chess

from ..data.utils import get_game_phase
from ..utils import get_move_id
from .utils import encode_position


_POSITION_CACHE: OrderedDict = OrderedDict()
_POSITION_CACHE_LOCK = threading.Lock()
_POSITION_CACHE_LIMIT = 2048


def _encoded_position_key(tokens, extra) -> bytes:
    digest = hashlib.blake2b(digest_size=16)
    digest.update(tokens.dtype.str.encode())
    digest.update(repr(tokens.shape).encode())
    digest.update(tokens.tobytes())
    for key in sorted(extra):
        value = extra[key]
        digest.update(key.encode())
        digest.update(value.dtype.str.encode())
        digest.update(repr(value.shape).encode())
        digest.update(value.tobytes())
    return digest.digest()


def _encode_position(board: chess.Board, encoding: str) -> dict:
    history = tuple(move.uci() for move in board.move_stack) if encoding == 'lc0' else ()
    cache_key = (encoding, board.fen(en_passant='fen'), history)
    with _POSITION_CACHE_LOCK:
        cached = _POSITION_CACHE.get(cache_key)
        if cached is not None:
            _POSITION_CACHE.move_to_end(cache_key)
            return cached

        position = encode_position(board, encoding)
        position['position_key'] = _encoded_position_key(position['tokens'], position['extra'])
        _POSITION_CACHE[cache_key] = position
        if len(_POSITION_CACHE) > _POSITION_CACHE_LIMIT:
            _POSITION_CACHE.popitem(last=False)
        return position


def _encode_routed_positions(board: chess.Board) -> dict:
    return {
        encoding: _encode_position(board, encoding)
        for encoding in _ROUTED_ENCODINGS
    }


def _model_move(
    board: chess.Board,
    model_id: str,
    game_id: int,
    play_mode: str,
    request_queue,
    response_queue,
) -> chess.Move:
    legal_moves = list(board.legal_moves)
    legal_ids = [get_move_id(move, board.turn) for move in legal_moves]
    is_consortium = model_id in _CONSORTIUM_NAMES
    request = {
        'model_id': model_id,
        'game_id': game_id,
        'play_mode': play_mode,
        'legal_ids': legal_ids,
    }

    if is_consortium:
        request['phase'] = get_game_phase(board)
        if play_mode == 'value':
            child_positions = []
            terminal_values = []
            child_phases = []
            for move in legal_moves:
                board.push(move)
                outcome = board.outcome(claim_draw=True)
                if outcome is None:
                    child_positions.append(_encode_routed_positions(board))
                    terminal_values.append(None)
                    child_phases.append(get_game_phase(board))
                else:
                    child_positions.append(None)
                    child_phases.append(None)
                    if outcome.winner is None:
                        terminal_values.append(0.5)
                    else:
                        terminal_values.append(1.0 if outcome.winner == board.turn else 0.0)
                board.pop()
            request['routed_child_positions'] = child_positions
            request['terminal_values'] = terminal_values
            request['child_phases'] = child_phases
        else:
            request['routed_positions'] = _encode_routed_positions(board)
    elif play_mode == 'value':
        child_positions = []
        terminal_values = []
        for move in legal_moves:
            board.push(move)
            outcome = board.outcome(claim_draw=True)
            if outcome is None:
                child_positions.append(_encode_position(board, _MODEL_ENCODINGS[model_id]))
                terminal_values.append(None)
            else:
                child_positions.append(None)
                if outcome.winner is None:
                    terminal_values.append(0.5)
                else:
                    terminal_values.append(1.0 if outcome.winner == board.turn else 0.0)
            board.pop()
        request['child_positions'] = child_positions
        request['terminal_values'] = terminal_values
    else:
        request.update(_encode_position(board, _MODEL_ENCODINGS[model_id]))

    request_queue.put(request)
    response = response_queue.get()
    if 'error' in response:
        raise RuntimeError(f'GPU inference failed: {response["error"]}')

    move_id = int(response['move_id'])
    for move in legal_moves:
        if get_move_id(move, board.turn) == move_id:
            return move
    raise RuntimeError(f'Model {model_id!r} returned illegal move id {move_id}')


_MODEL_ENCODINGS: dict[str, str] = {}
_ROUTED_ENCODINGS: tuple[str, ...] = ()
_CONSORTIUM_NAMES: frozenset[str] = frozenset()


def _play_game(
    start_fen: str,
    white_model: str,
    black_model: str,
    game_id: int,
    play_mode: str,
    max_game_plies: int,
    request_queue,
    response_queue,
) -> dict:
    board = chess.Board(start_fen, chess960=True)
    moves = []
    while len(moves) < max_game_plies:
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            termination = outcome.termination.name.lower()
            result = outcome.result()
            break

        model_id = white_model if board.turn == chess.WHITE else black_model
        move = _model_move(
            board, model_id, game_id, play_mode, request_queue, response_queue
        )
        moves.append(board.san(move))
        board.push(move)
    else:
        termination = 'max_game_plies'
        result = '1/2-1/2'

    if result == '1-0':
        white_score = 1.0
    elif result == '0-1':
        white_score = 0.0
    else:
        white_score = 0.5

    return {
        'game_id': game_id,
        'white_model': white_model,
        'black_model': black_model,
        'result': result,
        'white_score': white_score,
        'termination': termination,
        'plies': len(moves),
        'moves': ' '.join(moves),
    }


def play_tournament_worker(
    match_specs: list[dict],
    play_mode: str,
    max_game_plies: int,
    model_encodings: dict[str, str],
    routed_encodings: tuple[str, ...],
    consortium_names: tuple[str, ...],
    request_queue,
    response_queues: dict,
    result_queue,
) -> None:
    """Persistent round worker; concurrently drives all assigned paired games."""
    global _MODEL_ENCODINGS, _ROUTED_ENCODINGS, _CONSORTIUM_NAMES
    _MODEL_ENCODINGS = model_encodings
    _ROUTED_ENCODINGS = routed_encodings
    _CONSORTIUM_NAMES = frozenset(consortium_names)

    pending = {}
    futures = {}
    for spec in match_specs:
        match_id = spec['match_id']
        pending[match_id] = {
            'spec': spec,
            'games': [None, None],
            'remaining': 2,
            'errors': [],
        }

    try:
        with ThreadPoolExecutor(max_workers=max(1, 2 * len(match_specs))) as executor:
            for spec in match_specs:
                match_id = spec['match_id']
                for game_index, white_model in enumerate((spec['model_a'], spec['model_b'])):
                    black_model = spec['model_b'] if white_model == spec['model_a'] else spec['model_a']
                    game_id = match_id * 2 + game_index
                    future = executor.submit(
                        _play_game,
                        spec['opening_fen'],
                        white_model,
                        black_model,
                        game_id,
                        play_mode,
                        max_game_plies,
                        request_queue,
                        response_queues[game_id],
                    )
                    futures[future] = (match_id, game_index)

            for future in as_completed(futures):
                match_id, game_index = futures[future]
                match = pending[match_id]
                try:
                    match['games'][game_index] = future.result()
                except BaseException:
                    match['errors'].append(traceback.format_exc())
                match['remaining'] -= 1
                if match['remaining'] != 0:
                    continue

                spec = match['spec']
                event = {
                    'match_id': match_id,
                    'round': spec['round'],
                    'model_a': spec['model_a'],
                    'model_b': spec['model_b'],
                    'opening_fen': spec['opening_fen'],
                }
                if match['errors']:
                    event['error'] = '\n'.join(match['errors'])
                else:
                    event['games'] = match['games']
                result_queue.put(event)
    except BaseException:
        error = traceback.format_exc()
        for match_id, match in pending.items():
            if match['remaining']:
                spec = match['spec']
                result_queue.put({
                    'match_id': match_id,
                    'round': spec['round'],
                    'model_a': spec['model_a'],
                    'model_b': spec['model_b'],
                    'opening_fen': spec['opening_fen'],
                    'error': error,
                })
