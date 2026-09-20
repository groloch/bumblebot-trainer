from dataclasses import dataclass


@dataclass
class SSLDataConfig:
    min_moves: int
    max_prediction_depth: int
    encoding: str
