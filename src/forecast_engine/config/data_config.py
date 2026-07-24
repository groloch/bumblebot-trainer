from dataclasses import dataclass


@dataclass
class DataConfig:
    tablebase_path: str
    fens_path: str
    datasets: list[str]


@dataclass
class SSLDataConfig:
    min_moves: int
    max_prediction_depth: int
