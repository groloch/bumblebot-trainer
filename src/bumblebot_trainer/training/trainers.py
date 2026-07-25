import os

import torch
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
import torch.nn.functional as F

from tqdm import tqdm

from . import Trainer
from .utils import build_model_config
from ..modeling.model import ChessModel
from ..modeling.heads import PolicyOutput, ValueOutput
from ..modeling.ssl import SSLChessModel, Predictor
from ..data import CombinedPositionDataset, LichessStandardGamesSSLDataset, SSLCollateFn
from ..tracking import AccumulationBuffer, binary_f1, multiclass_f1

from ..config import (
    TrainingConfig,
    TrackingConfig,
    DataConfig,
    SSLTrainingConfig,
    EncoderConfig,
    CFEncoderConfig,
    PredictorConfig,
    SSLDataConfig,
    SSLModelConfig
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


class SSLTrainer(Trainer):
    """Trainer for SSL pipeline, inspired by V-JEPA 2-AC
    """
    def __init__(self, config, config_path):
        super().__init__(config, config_path)

        # type hints
        self.teacher: torch.nn.Module
        self.predictor: torch.nn.Module
        self.training_config: SSLTrainingConfig
        self.model_config: SSLModelConfig
        self.data_config: SSLDataConfig
        # \

    def _build_configs(self, config):
        self.training_config = SSLTrainingConfig(**config['training'])

        hidden_size = config['model']['hidden_size']
        encoder_config = CFEncoderConfig(hidden_size=hidden_size, **config['model'].pop('encoder'))

        self.model_config = SSLModelConfig(encoder_config=encoder_config, **config['model'])

        self.predictor_config = PredictorConfig(hidden_size=self.model_config.hidden_size, **config['predictor'])
        self.tracking_config = TrackingConfig(**config['tracking'])
        self.data_config = SSLDataConfig(**config['data'])

    def _build_model(self):
        self.model = SSLChessModel(self.model_config)
        self.teacher = AveragedModel(self.model, multi_avg_fn=get_ema_multi_avg_fn(self.training_config.ema_decay))
        self.predictor = Predictor(self.predictor_config)

        for p in self.teacher.parameters():
            p.requires_grad = False

    def _load_datasets(self):
        dataset = LichessStandardGamesSSLDataset(
            min_moves=self.data_config.min_moves,
            max_prediction_depth=self.data_config.max_prediction_depth,
            encoding=self.data_config.encoding,
        )

        print(f'{dataset.__class__.__name__} size: {len(dataset):,}')

        self.train_dataloader = DataLoader(
            dataset,
            batch_size=self.training_config.batch_size,
            shuffle=True,
            num_workers=self.training_config.num_workers,
            pin_memory=True,
            collate_fn=SSLCollateFn(self.data_config.max_prediction_depth),
        )

    def run(self):
        self.model.train()
        self.model.to(self.device)

        self.predictor.train()
        self.predictor.to(self.device)

        self.teacher.eval()
        self.teacher.to(self.device)

        self.optimizer.zero_grad(set_to_none=True)

        gradient_accumulation_steps = self.training_config.gradient_accumulation_steps

        step = 0
        total_steps = (len(self.train_dataloader)+gradient_accumulation_steps-1) // gradient_accumulation_steps
        total_steps = min(
            self.training_config.max_steps,
            total_steps
        )
        pbar = tqdm(total=total_steps, desc='SSL Training')

        acc_buffer = AccumulationBuffer(gradient_accumulation_steps, self.device)

        for partial_step, batch in enumerate(self.train_dataloader, start=1):
            tokens, tokens_, legal_moves, attacks, moves, moves_attention_mask = batch

            tokens = tokens.to(self.device)
            tokens_ = tokens_.to(self.device)

            legal_moves = legal_moves.to(self.device, dtype=torch.float32)
            attacks = attacks.to(self.device, dtype=torch.long)
            moves = moves.to(self.device)
            moves_attention_mask = moves_attention_mask.to(self.device)

            # jepa-like forward pass
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                with torch.no_grad():
                    target_embed, _, _ = self.teacher.module(tokens_)

                student_embed, logits, losses = self.model(
                    tokens,
                    target={'legal': legal_moves, 'attacks': attacks}
                )
                prediction = self.predictor(student_embed, moves, moves_attention_mask)

                legal_loss = losses['legal']
                legal_logits = logits['legal']

                attacks_loss = losses['attacks']
                attacks_logits = logits['attacks']

                ssl_loss = F.smooth_l1_loss(prediction, target_embed.detach(), beta=0.1)

                total_loss = ssl_loss / gradient_accumulation_steps * self.training_config.ssl_loss_weight
                total_loss += legal_loss / gradient_accumulation_steps * self.training_config.legal_loss_weight
                total_loss += attacks_loss / gradient_accumulation_steps * self.training_config.attacks_loss_weight

            total_loss.backward()

            acc_buffer.update('ssl_loss_unscaled', ssl_loss.detach(), partial_step)
            acc_buffer.update('ssl_loss', ssl_loss.detach() * self.training_config.ssl_loss_weight, partial_step)

            acc_buffer.update('legal_loss_unscaled', legal_loss.detach(), partial_step)
            acc_buffer.update('legal_loss', legal_loss.detach() * self.training_config.legal_loss_weight, partial_step)
            acc_buffer.update('legal_f1', binary_f1(legal_logits.detach(), legal_moves), partial_step)

            acc_buffer.update('attacks_loss_unscaled', attacks_loss.detach(), partial_step)
            acc_buffer.update('attacks_loss', attacks_loss.detach() * self.training_config.attacks_loss_weight, partial_step)
            acc_buffer.update('attacks_f1', multiclass_f1(attacks_logits.detach(), attacks), partial_step)

            if (partial_step + 1) % gradient_accumulation_steps == 0 or partial_step == len(self.train_dataloader):
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.parameters()) + list(self.predictor.parameters()),
                    max_norm=self.training_config.max_grad_norm,
                )

                self.optimizer.step()
                self.scheduler.step()

                self.teacher.update_parameters(self.model)

                self.logger.update('lr', self.scheduler.get_last_lr()[0])

                self.logger.update('ssl_loss_unscaled', acc_buffer.get_mean('ssl_loss_unscaled'))
                self.logger.update('ssl_loss', acc_buffer.get_mean('ssl_loss'))

                self.logger.update('legal_loss_unscaled', acc_buffer.get_mean('legal_loss_unscaled'))
                self.logger.update('legal_loss', acc_buffer.get_mean('legal_loss'))
                self.logger.update('legal_f1', acc_buffer.get_mean('legal_f1'))

                self.logger.update('attacks_loss_unscaled', acc_buffer.get_mean('attacks_loss_unscaled'))
                self.logger.update('attacks_loss', acc_buffer.get_mean('attacks_loss'))
                self.logger.update('attacks_f1', acc_buffer.get_mean('attacks_f1'))

                acc_buffer.reset()
                pbar.update(1)

                self.optimizer.zero_grad(set_to_none=True)

                step += 1
                if step >= self.training_config.max_steps:
                    break

                if step >= 100 and step % 10 == 0:
                    pbar.set_description(
                        f'SSL Training | {self.logger.log(step, exclude_if_contains=['unscaled'])}'
                    )

                if step % self.training_config.save_every == 0:
                    self.ckpt(
                        step=step,
                        training_state=True,
                        additional_modules={'predictor': self.predictor, 'teacher': self.teacher}
                    )
