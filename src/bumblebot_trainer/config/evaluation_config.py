from dataclasses import dataclass


@dataclass
class EvaluationConfig:
    every: int
    batch_size: int
    num_workers: int
    num_puzzles: int
    elo_min: float
    elo_max: float
    elo_step: float
    prior_elo_mean: float = 1500.0
    prior_elo_std: float = 1000.0
