from dataclasses import dataclass


@dataclass
class TrainingConfig:
    max_steps: int
    batch_size: int
    learning_rate: float
    warmup_steps: int
    weight_decay: float
    max_grad_norm: float
    seed: int
    gradient_accumulation_steps: int
    forecast_depth: int
    num_workers: int
    forecast_loss_weight: float
    logdir: str
    save_every: int
    name: str

@dataclass
class TuningConfig(TrainingConfig):
    pretrained_logdir: str
    tune_value: bool
    tune_policy: bool
    tune_forecast: bool


@dataclass
class SSLTrainingConfig:
    max_steps: int
    batch_size: int
    learning_rate: float
    warmup_steps: int
    weight_decay: float
    max_grad_norm: float
    seed: int
    gradient_accumulation_steps: int
    num_workers: int
    logdir: str
    save_every: int
    ema_decay: float
    name: str
    legal_loss_weight: float
    attacks_loss_weight: float
    ssl_loss_weight: float
    perceptive_loss_weight: float
