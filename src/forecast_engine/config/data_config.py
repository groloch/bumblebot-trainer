from dataclasses import dataclass


@dataclass
class DataConfig:
    datasets: list[str]
    encoding: str


@dataclass
class SSLDataConfig:
    min_moves: int
    max_prediction_depth: int
    encoding: str
