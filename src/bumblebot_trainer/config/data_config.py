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


@dataclass
class Lc0DataConfig:
    directory: str
    encoding: str
