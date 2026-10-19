from dataclasses import dataclass


@dataclass
class StockfishEvaluationConfig:
    stockfish_path: str
    num_game_pairs: int
    stockfish_elo_ranges: list[list[int]]
    num_workers: int
    batch_size: int
    batch_timeout_ms: float
    move_time_seconds: float
    max_game_plies: int
    seed: int
    prior_elo_mean: float
    prior_elo_std: float
    elo_min: float
    elo_max: float
    elo_step: float
    book_path: str | None = None
    results_path: str | None = None
    encoding: str | None = None
    onnx_tokens_input: str | None = None
    onnx_legal_input: str | None = None
    onnx_policy_output: str | None = None
    consortium: dict | None = None
