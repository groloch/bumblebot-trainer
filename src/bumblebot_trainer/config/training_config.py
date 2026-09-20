from dataclasses import dataclass


@dataclass
class DropoutScheduleConfig:
    min_dropout: float
    max_dropout: float
    convergence_rate: float
    f1_threshold: float
    min_steps_between_updates: int


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
    num_workers: int
    logdir: str
    save_every: int
    ema_decay: float
    name: str
    legal_loss_weight: float
    attacks_loss_weight: float
    ssl_loss_weight: float
    perceptive_loss_weight: float
    dropout_schedule: DropoutScheduleConfig
