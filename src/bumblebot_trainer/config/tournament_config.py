from dataclasses import dataclass


@dataclass
class TournamentConfig:
    rounds: int
    play_mode: str
    num_workers: int
    max_active_matches: int
    batch_size: int
    value_batch_size: int
    model_output_cache_size: int
    batch_timeout_ms: float
    max_game_plies: int
    seed: int
    prior_elo_mean: float
    prior_elo_std: float
    elo_min: float
    elo_max: float
    models: list[dict | str] | None = None
    pin_player: dict | None = None
    consortium: dict | None = None
    consortiums: list[dict] | None = None
    router: dict | None = None
    book_path: str | None = None
    results_path: str | None = None
