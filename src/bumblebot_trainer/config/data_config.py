from dataclasses import dataclass


@dataclass
class SSLDataConfig:
    min_moves: int
    max_prediction_depth: int
    encoding: str
    dataset_mix: dict[str, float]
    streaming: bool
    min_elo: int
    shuffle_buffer_size: int = 128

    def __post_init__(self):
        if self.shuffle_buffer_size < 1:
            raise ValueError("shuffle_buffer_size must be positive")
