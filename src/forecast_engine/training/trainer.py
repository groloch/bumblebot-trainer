import os
import math

import torch
from torch.optim import AdamW
import mlflow

from .utils import init_logdir, init_run, model_parameters
from ..config import TrainingConfig, TrackingConfig
from ..tracking import MetricLogger


class Trainer:
    """Trainer abstract class. To implement a trainer, you must override the following function
    - _build_configs(config: dict): build the training, model, data and tracking configs from the config dict
    - _build_model(): build the model to train
    - _load_datasets(): load the datasets and create the dataloaders
    - run(): implement the training loop

    """
    def __init__(self, config, config_path):
        self.training_config: TrainingConfig = ...
        self.model_config = ...
        self.data_config = ...
        self.tracking_config: TrackingConfig = ...

        print('Loading run configuration')
        self._build_configs(config)

        self.logdir = init_logdir(self.training_config.logdir, config_path)
        print(f'Logging to {self.logdir}')

        init_run(self.training_config.seed, self.tracking_config)

        print('Building model')
        self.model: torch.nn.Module = ...
        self._build_model()
        total_params, trainable_params = model_parameters(self.model)
        print(f'Model Parameters: {total_params/1e6:.2f}M total, {trainable_params/1e6:.2f}M trainable')

        print('Loading datasets')
        self.train_dataloader = ...
        self._load_datasets()

        print('Building optimizer and scheduler')
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay
        )

        def lr_lambda(current_step):
            if current_step < self.training_config.warmup_steps:
                return float(current_step) / float(max(1, self.training_config.warmup_steps))
            decay_start = self.training_config.max_steps * 0.9
            if current_step < decay_start:
                return 1.0
            progress = (current_step - decay_start) / (self.training_config.max_steps - decay_start)
            return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        self.logger = MetricLogger(self.tracking_config.use_mlflow)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _build_configs(self, config: dict):
        raise NotImplementedError()

    def _build_model(self):
        raise NotImplementedError()

    def _load_datasets(self):
        raise NotImplementedError()


    def run(self):
        raise NotImplementedError()

    def ckpt(
            self,
            step: int | None = None,
            training_state: bool = True,
            additional_modules: dict[str, torch.nn.Module] | None = None):
        optimizer_name = 'optimizer.pth'
        scheduler_name = 'scheduler.pth'
        if step is None:
            model_name = f'{self.training_config.name}.pth'
        else:
            model_name = f'{self.training_config.name}_step{step}.pth'
        torch.save(self.model.state_dict(), os.path.join(self.logdir, model_name))

        if training_state:
            torch.save(self.optimizer.state_dict(), os.path.join(self.logdir, optimizer_name))
            torch.save(self.scheduler.state_dict(), os.path.join(self.logdir, scheduler_name))

        if additional_modules is not None:
            for name, module in additional_modules.items():
                torch.save(module.state_dict(), os.path.join(self.logdir, f'{name}.pth'))

    def wrapup(self):
        if self.tracking_config.use_mlflow:
            mlflow.end_run()
        self.ckpt(training_state=False)
