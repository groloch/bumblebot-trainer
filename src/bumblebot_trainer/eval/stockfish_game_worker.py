"""Game-pair worker subprocess used by the standalone Stockfish evaluator."""

import traceback
from concurrent.futures import ThreadPoolExecutor

import chess
import chess.engine

from ..data.utils import encode_board
from ..utils import get_move_id


def _configure_stockfish(stockfish_path: str, elo: int) -> chess.engine.SimpleEngine:
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    try:
        missing = {'Threads', 'UCI_LimitStrength', 'UCI_Elo'} - set(engine.options)
        if missing:
            raise RuntimeError(
                f'{stockfish_path} does not expose required UCI options: {sorted(missing)}'
            )
        elo_option = engine.options['UCI_Elo']
        if elo_option.min is not None and elo < elo_option.min:
            raise ValueError(f'Stockfish Elo {elo} is below its minimum {elo_option.min}')
        if elo_option.max is not None and elo > elo_option.max:
            raise ValueError(f'Stockfish Elo {elo} is above its maximum {elo_option.max}')
        engine.configure({'Threads': 1, 'UCI_LimitStrength': True, 'UCI_Elo': elo})
        return engine
    except BaseException:
        engine.quit()
        raise


def _request_model_move(
    board: chess.Board,
    encoding: str,
    game_id: int,
    request_queue,
    response_queue,
) -> chess.Move:
    tokens, extra = encode_board(board, encoding)
    legal_moves = list(board.legal_moves)
    legal_ids = [get_move_id(move, board.turn) for move in legal_moves]

    request_queue.put({
        'game_id': game_id,
        'tokens': tokens.numpy(),
        'extra': {key: value.numpy() for key, value in extra.items()},
        'legal_ids': legal_ids,
    })
    response = response_queue.get()
    if 'error' in response:
        raise RuntimeError(f'GPU model evaluation failed: {response["error"]}')

    move_id = int(response['move_id'])
    for move in legal_moves:
        if get_move_id(move, board.turn) == move_id:
            return move
    raise RuntimeError(f'Model returned non-legal move id {move_id}')


def _play_game(
    start_fen: str,
    model_color: chess.Color,
    stockfish_elo: int,
    game_id: int,
    encoding: str,
    stockfish_path: str,
    move_time_seconds: float,
    max_game_plies: int,
    request_queue,
    response_queue,
) -> dict:
    board = chess.Board(start_fen, chess960=True)
    moves = []
    engine = _configure_stockfish(stockfish_path, stockfish_elo)
    try:
        termination = 'unknown'
        while len(moves) < max_game_plies:
            outcome = board.outcome(claim_draw=True)
            if outcome is not None:
                termination = outcome.termination.name.lower()
                result = outcome.result()
                break

            if board.turn == model_color:
                move = _request_model_move(
                    board, encoding, game_id, request_queue, response_queue
                )
            else:
                move = engine.play(board, chess.engine.Limit(time=move_time_seconds)).move
                if move is None or move not in board.legal_moves:
                    raise RuntimeError('Stockfish did not return a legal move')

            moves.append(board.san(move))
            board.push(move)
        else:
            termination = 'max_game_plies'
            result = '1/2-1/2'
    finally:
        engine.quit()

    if result == '1/2-1/2':
        score = 0.5
    else:
        winner = chess.WHITE if result == '1-0' else chess.BLACK
        score = 1.0 if winner == model_color else 0.0

    return {
        'model_color': 'white' if model_color == chess.WHITE else 'black',
        'result': result,
        'score': score,
        'termination': termination,
        'plies': len(moves),
        'moves': ' '.join(moves),
    }


def play_game_pair(
    pair_id: int,
    start_fen: str,
    stockfish_elo: int,
    encoding: str,
    stockfish_path: str,
    move_time_seconds: float,
    max_game_plies: int,
    request_queue,
    response_queues: tuple,
    result_queue,
):
    try:
        # Play both colors from the exact same opening. Each game owns a Stockfish process.
        with ThreadPoolExecutor(max_workers=2) as executor:
            games = [
                executor.submit(
                    _play_game,
                    start_fen,
                    model_color,
                    stockfish_elo,
                    pair_id * 2 + game_index,
                    encoding,
                    stockfish_path,
                    move_time_seconds,
                    max_game_plies,
                    request_queue,
                    response_queues[game_index],
                )
                for game_index, model_color in enumerate((chess.WHITE, chess.BLACK))
            ]
            game_results = [future.result() for future in games]

        result_queue.put({
            'pair_id': pair_id,
            'opening_fen': start_fen,
            'stockfish_elo': stockfish_elo,
            'games': game_results,
        })
    except BaseException:
        result_queue.put({
            'pair_id': pair_id,
            'error': traceback.format_exc(),
        })
