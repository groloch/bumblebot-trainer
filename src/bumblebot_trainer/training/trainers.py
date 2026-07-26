import itertools
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import Trainer
from .utils import build_model_config
from ..modeling.model import ChessModel
from ..modeling.heads import PolicyOutput, ValueOutput
from ..data import CombinedPositionDataset
from ..tracking import AccumulationBuffer

from ..config import (
    TrainingConfig,
    TrackingConfig,
    DataConfig,
)


class PVTrainer(Trainer):
    """Trainer for a policy/value model
    """
    def _build_configs(self, config):
        self.training_config = TrainingConfig(**config['training'])
        self.model_config = build_model_config(config['model'])
        self.tracking_config = TrackingConfig(**config['tracking'])
        self.data_config = DataConfig(**config['data'])

    def _build_model(self):
        self.model = ChessModel(self.model_config)

    def _load_datasets(self):
        datasets = []
        # TODO load datasets

        dataset = CombinedPositionDataset(datasets)
        print('Data splits sizes:')
        for dataset_ in datasets:
            print(f'{dataset_.__class__.__name__} size: {len(dataset_):,}')
        print(f'Total: {len(dataset):,}')

        self.train_dataloader = DataLoader(
            dataset,
            batch_size=self.training_config.batch_size,
            shuffle=True,
            num_workers=self.training_config.num_workers,
            pin_memory=True
        )

        raise NotImplementedError('No dataset implemented yet.')

    def run(self):
        self.model.train()
        self.model.to(self.device, dtype=torch.bfloat16)

        self.optimizer.zero_grad(set_to_none=True)

        gradient_accumulation_steps = self.training_config.gradient_accumulation_steps

        step = 0
        total_steps = (len(self.train_dataloader)+gradient_accumulation_steps-1) // gradient_accumulation_steps
        total_steps = min(
            self.training_config.max_steps,
            total_steps
        )

        pbar = tqdm(total=total_steps, desc='Training')

        acc_buffer = AccumulationBuffer(gradient_accumulation_steps, self.device)

        # type hints
        policy_out: PolicyOutput
        value_out: ValueOutput
        tokens: torch.Tensor
        target_dict: dict[str, torch.Tensor]
        # \

        for partial_step, batch in enumerate(self.train_dataloader, start=1):
            tokens, target_dict = batch

            tokens = tokens.to(self.device)
            target = {
                k: v.to(self.device, dtype=torch.bfloat16)
                for k, v in target_dict.items()
            }

            _, policy_out, value_out = self.model(
                tokens,
                target
            )

            total_loss = policy_out.loss
            acc_buffer.update('policy_loss', policy_out.loss.detach(), partial_step)

            if value_out.loss is not None:
                total_loss += value_out.loss
                acc_buffer.update('value_loss', value_out.loss.detach(), partial_step)

            acc_buffer.update('train_loss', total_loss.detach(), partial_step)

            accuracy = (policy_out.logits.argmax(dim=-1) == target['policy'].argmax(dim=-1)).float().mean()
            acc_buffer.update('accuracy', accuracy.detach(), partial_step)

            total_loss = total_loss / gradient_accumulation_steps
            total_loss.backward()

            if partial_step % gradient_accumulation_steps == 0 or partial_step == len(self.train_dataloader):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.training_config.max_grad_norm)

                self.optimizer.step()
                self.scheduler.step()

                self.logger.update('train_loss', acc_buffer.get_mean('train_loss'))
                self.logger.update('policy_loss', acc_buffer.get_mean('policy_loss'))
                if value_out.loss is not None:
                    self.logger.update('value_loss', acc_buffer.get_mean('value_loss', ignore_zeros=True))
                self.logger.update('lr', self.scheduler.get_last_lr()[0])
                self.logger.update('accuracy', acc_buffer.get_mean('accuracy'))

                acc_buffer.reset()
                pbar.update(1)

                self.optimizer.zero_grad(set_to_none=True)

                step += 1
                if step >= self.training_config.max_steps:
                    break

                if step >= 100 and step % 10 == 0:
                    pbar.set_description(
                        f'Training | {self.logger.log(step)}'
                    )

                if step % self.training_config.save_every == 0:
                    self.ckpt(step=step, training_state=True)
