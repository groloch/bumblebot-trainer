from dataclasses import dataclass


@dataclass
class EncoderConfig:
    hidden_size: int
    num_layers: int
    intermediate_size: int
    num_heads: int


@dataclass
class PredictorConfig(EncoderConfig):
    pass

@dataclass
class EmbeddingCompressorConfig:
    hidden_size: int


@dataclass
class DraftModelConfig:
    hidden_size: int


@dataclass
class ModelConfig:
    input_size: int
    hidden_size: int
    intermediate_size: int
    encoder_name: str
    encoder_config: EncoderConfig = None


@dataclass
class SSLModelConfig:
    input_size: int
    hidden_size: int
    intermediate_size: int
    encoder_name: str
    encoder_config: EncoderConfig = None
