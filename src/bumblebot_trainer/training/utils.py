import sys
import os
import shutil
import random

import torch
import numpy as np
import mlflow

from ..config import TrackingConfig, ModelConfig, EncoderConfig


def init_logdir(logdir: str, config_path: str) -> str:
    i = 0
    proposed_logdir = os.path.join(logdir, f'run_{i}')
    while os.path.exists(proposed_logdir):
        i += 1
        proposed_logdir = os.path.join(logdir, f'run_{i}')
    os.makedirs(proposed_logdir)
    shutil.copy2(config_path, proposed_logdir)
    return proposed_logdir

def init_run(seed, tracking_config: TrackingConfig):
    if not(torch.cuda.is_available()):
        print("CUDA is not available. Exiting.")
        sys.exit(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if tracking_config.use_mlflow:
        mlflow.set_tracking_uri(tracking_config.tracking_uri)
        mlflow.set_experiment(tracking_config.experiment_name)
        mlflow.start_run()

def model_parameters(model: torch.nn.Module):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total_params, trainable_params

def build_model_config(config: dict) -> ModelConfig:
    hidden_size = config['hidden_size']
    encoder_config = EncoderConfig(hidden_size=hidden_size, **config.pop('encoder'))
    return ModelConfig(
        encoder_config=encoder_config,
        **config
    )