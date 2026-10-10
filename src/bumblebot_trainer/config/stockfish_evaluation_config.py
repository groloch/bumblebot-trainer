from dataclasses import dataclass


@dataclass
class StockfishEvaluationConfig:
    stockfish_path: str
    book_path: str
    num_game_pairs: int
    stockfish_elo_ranges: list[list[int]]
    num_workers: int = 4
    batch_size: int = 64
    batch_timeout_ms: float = 5.0
    move_time_seconds: float = 0.1
    max_game_plies: int = 400
    seed: int = 42
    results_path: str | None = None
    # Required for standalone ONNX checkpoints; logdir mode reads encoding from its run config.
    encoding: str | None = None
    onnx_tokens_input: str | None = None
    onnx_legal_input: str | None = None
    onnx_policy_output: str | None = None
    prior_elo_mean: float = 1500.0
    prior_elo_std: float = 1000.0
    elo_min: float = 200.0
    elo_max: float = 3800.0
    elo_step: float = 2.0
