from dataclasses import dataclass


@dataclass
class TournamentConfig:
    rounds: int
    play_mode: str = 'policy'
    models: list[dict | str] | None = None
    consortium: dict | None = None
    consortiums: list[dict] | None = None
    router: dict | None = None
    book_path: str | None = None
    num_workers: int = 4
    max_active_matches: int = 128
    batch_size: int = 128
    value_batch_size: int = 64
    model_output_cache_size: int = 256
    batch_timeout_ms: float = 5.0
    max_game_plies: int = 400
    seed: int = 42
    results_path: str | None = None
    prior_elo_mean: float = 1500.0
    prior_elo_std: float = 1000.0
    elo_min: float = 200.0
    elo_max: float = 3800.0
