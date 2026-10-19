from dataclasses import dataclass


@dataclass
class SSLDataConfig:
    min_moves: int
    max_prediction_depth: int
    encoding: str
    dataset_mix: dict[str, float]
    streaming: bool
    min_elo: int
