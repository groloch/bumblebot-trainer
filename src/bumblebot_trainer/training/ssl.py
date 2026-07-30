import os
import itertools

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import AdamW, Muon
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from . import Trainer
from .utils import wsd_schedule, dropout_schedule
from ..modeling import SSLChessModel, Predictor
from ..data import LichessStandardGamesSSLDataset, SSLCollator, Lc0PositionDataset
from ..tracking import AccumulationBuffer
from ..tracking.metrics import (
    binary_f1_from_stats,
    binary_f1_stats,
    multiclass_confusion_matrix,
    multiclass_f1_from_confusion_matrix,
)
from ..modeling.utils import ssl_to_pv_model
from ..config import (
    SSLTrainingConfig,
    SSLModelConfig,
    CFEncoderConfig,
    SSLDataConfig,
    PredictorConfig,
    TrackingConfig,
    PVTuningConfig,
    Lc0DataConfig,
    DropoutScheduleConfig,
)


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

        self.muon_optimizer: Muon
        self.adamw_optimizer: AdamW
        self.muon_scheduler: torch.optim.lr_scheduler.LambdaLR
        self.adamw_scheduler: torch.optim.lr_scheduler.LambdaLR
        # \ type hints

        self._build_optimizer()

    def _build_optimizer(self):
        muon_params = [
            p for n, p in itertools.chain(
                self.model.named_parameters(),
                self.predictor.named_parameters()
            ) if p.requires_grad and
            p.ndim == 2 and
            not any(nd in n for nd in ['head', 'embed'])
        ]
        muon_param_ids = {id(p) for p in muon_params}
        adamw_params = [
            p for n, p in itertools.chain(
                self.model.named_parameters(),
                self.predictor.named_parameters()
            ) if p.requires_grad and
            id(p) not in muon_param_ids
        ]

        self.muon_optimizer = Muon(
            muon_params,
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
            adjust_lr_fn='match_rms_adamw'
        )
        self.adamw_optimizer = AdamW(
            adamw_params,
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
            betas=(0.9, 0.98),
            eps=1e-7
        )

        lr_schedule_fn = wsd_schedule(
            warmup_steps=self.training_config.warmup_steps,
            max_steps=self.training_config.max_steps
        )
        self.muon_scheduler = LambdaLR(self.muon_optimizer, lr_schedule_fn)
        self.adamw_scheduler = LambdaLR(self.adamw_optimizer, lr_schedule_fn)

        del self.optimizer
        del self.scheduler

    def ckpt(self, step=None, training_state=True, additional_modules=None):
        if training_state:
            optimizer_name = 'muon_optimizer.pth'
            adamw_optimizer_name = 'adamw_optimizer.pth'
            muon_scheduler_name = 'muon_scheduler.pth'
            adamw_scheduler_name = 'adamw_scheduler.pth'
            torch.save(self.muon_optimizer.state_dict(), os.path.join(self.logdir, optimizer_name))
            torch.save(self.adamw_optimizer.state_dict(), os.path.join(self.logdir, adamw_optimizer_name))
            torch.save(self.muon_scheduler.state_dict(), os.path.join(self.logdir, muon_scheduler_name))
            torch.save(self.adamw_scheduler.state_dict(), os.path.join(self.logdir, adamw_scheduler_name))
        super().ckpt(step=step, training_state=False, additional_modules=additional_modules)

    def _optimizer_step(self):
        self.muon_optimizer.step()
        self.adamw_optimizer.step()

        self.muon_scheduler.step()
        self.adamw_scheduler.step()

    def _optimizer_zero_grad(self, set_to_none=True):
        self.muon_optimizer.zero_grad(set_to_none=set_to_none)
        self.adamw_optimizer.zero_grad(set_to_none=set_to_none)

    def _build_configs(self, config):
        dropout_schedule_cfg = DropoutScheduleConfig(**config['training'].pop('dropout_schedule'))
        self.training_config = SSLTrainingConfig(dropout_schedule=dropout_schedule_cfg, **config['training'])

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

    def _update_dropout(self, perceptive_legal_f1: float):
        ds = self.training_config.dropout_schedule
        self._steps_since_dropout_update += 1
        if (perceptive_legal_f1 >= ds.f1_threshold
                and self._steps_since_dropout_update >= ds.min_steps_between_updates):
            self._n_dropout_updates += 1
            self._steps_since_dropout_update = 0
            self._current_dropout = self._dropout_fn(self._n_dropout_updates)

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
            collate_fn=SSLCollator(self.data_config.max_prediction_depth)
        )

    def run(self):
        self.model.train()
        self.model.to(self.device)

        self.predictor.train()
        self.predictor.to(self.device)

        self.teacher.eval()
        self.teacher.to(self.device)

        self._optimizer_zero_grad(set_to_none=True)

        gradient_accumulation_steps = self.training_config.gradient_accumulation_steps
        _ds = self.training_config.dropout_schedule
        self._dropout_fn = dropout_schedule(
            _ds.min_dropout,
            _ds.max_dropout,
            _ds.convergence_rate,
        )
        self._current_dropout = self._dropout_fn(0)
        self._n_dropout_updates = 0
        self._steps_since_dropout_update = 0

        step = 0
        total_steps = (len(self.train_dataloader)+gradient_accumulation_steps-1) // gradient_accumulation_steps
        total_steps = min(
            self.training_config.max_steps,
            total_steps
        )
        pbar = tqdm(total=total_steps, desc='SSL Training')

        acc_buffer = AccumulationBuffer(gradient_accumulation_steps, self.device)
        attacks_num_classes = self.model.attacks_head.output_dim
        legal_f1_stats = torch.zeros(3, device=self.device)
        attacks_confusion = torch.zeros((attacks_num_classes, attacks_num_classes), device=self.device)
        perceptive_legal_f1_stats = torch.zeros(3, device=self.device)
        perceptive_attacks_confusion = torch.zeros(
            (attacks_num_classes, attacks_num_classes),
            device=self.device
        )

        for partial_step, batch in enumerate(self.train_dataloader):
            tokens, tokens_, legal_moves, attacks, legal_moves_, attacks_, moves, moves_attention_mask = batch

            tokens = tokens.to(self.device, non_blocking=True)
            tokens_ = tokens_.to(self.device, non_blocking=True)

            legal_moves = legal_moves.to(self.device, dtype=torch.float32, non_blocking=True)
            attacks = attacks.to(self.device, dtype=torch.long, non_blocking=True)

            legal_moves_ = legal_moves_.to(self.device, dtype=torch.float32, non_blocking=True)
            attacks_ = attacks_.to(self.device, dtype=torch.long, non_blocking=True)

            moves = moves.to(self.device, non_blocking=True)
            moves_attention_mask = moves_attention_mask.to(self.device, non_blocking=True)

            # jepa-like forward pass
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                with torch.no_grad():
                    _, target_embed = self.teacher.module.embed(tokens_)

                student_embed, logits, losses = self.model(
                    tokens,
                    target={'legal': legal_moves, 'attacks': attacks}
                )
                pred_raw, pred_norm = self.predictor(
                    student_embed,
                    moves,
                    moves_attention_mask,
                    self._current_dropout
                )

                legal_logits = logits['legal']
                legal_loss = losses['legal']

                attacks_logits = logits['attacks']
                attacks_loss = losses['attacks']

                ssl_loss = F.smooth_l1_loss(pred_norm, target_embed.detach(), beta=0.1)

                perceptive_logits, perceptive_losses = self.teacher.module.heads_out(
                    pred_raw,
                    target={'legal': legal_moves_, 'attacks': attacks_}
                )

                perceptive_legal_loss = perceptive_losses['legal']
                perceptive_legal_logits = perceptive_logits['legal']

                perceptive_attacks_loss = perceptive_losses['attacks']
                perceptive_attacks_logits = perceptive_logits['attacks']

                total_loss = ssl_loss / gradient_accumulation_steps * self.training_config.ssl_loss_weight

                total_loss += legal_loss / gradient_accumulation_steps * self.training_config.legal_loss_weight
                total_loss += attacks_loss / gradient_accumulation_steps * self.training_config.attacks_loss_weight

                total_loss += perceptive_legal_loss / gradient_accumulation_steps * \
                    self.training_config.legal_loss_weight * self.training_config.perceptive_loss_weight
                total_loss += perceptive_attacks_loss / gradient_accumulation_steps * \
                    self.training_config.attacks_loss_weight * self.training_config.perceptive_loss_weight

            total_loss.backward()

            # metrics
            # all metrics contain a sliding window average over the last 100 steps
            # all losses contain a unscaled version to compare between runs
            acc_buffer.update('ssl_loss_unscaled', ssl_loss.detach(), partial_step)
            acc_buffer.update(
                'ssl_loss',
                ssl_loss.detach() * self.training_config.ssl_loss_weight,
                partial_step
            )

            acc_buffer.update('legal_loss_unscaled', legal_loss.detach(), partial_step)
            acc_buffer.update(
                'legal_loss',
                legal_loss.detach() * self.training_config.legal_loss_weight,
                partial_step
            )
            legal_f1_stats += binary_f1_stats(legal_logits.detach(), legal_moves)

            acc_buffer.update('attacks_loss_unscaled', attacks_loss.detach(), partial_step)
            acc_buffer.update(
                'attacks_loss',
                attacks_loss.detach() * self.training_config.attacks_loss_weight,
                partial_step
            )
            attacks_confusion += multiclass_confusion_matrix(attacks_logits.detach(), attacks)

            acc_buffer.update(
                'perceptive_legal_loss_unscaled', perceptive_legal_loss.detach(),
                partial_step
            )
            acc_buffer.update(
                'perceptive_legal_loss', perceptive_legal_loss.detach() * \
                    self.training_config.legal_loss_weight * self.training_config.perceptive_loss_weight,
                partial_step
            )
            perceptive_legal_f1_stats += binary_f1_stats(
                perceptive_legal_logits.detach(),
                legal_moves_
            )

            acc_buffer.update(
                'perceptive_attacks_loss_unscaled', perceptive_attacks_loss.detach(),
                partial_step
            )
            acc_buffer.update(
                'perceptive_attacks_loss',
                perceptive_attacks_loss.detach() * self.training_config.attacks_loss_weight * \
                    self.training_config.perceptive_loss_weight,
                partial_step
            )
            perceptive_attacks_confusion += multiclass_confusion_matrix(
                perceptive_attacks_logits.detach(),
                attacks_
            )
            # \ metrics

            if (partial_step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.parameters()) + list(self.predictor.parameters()),
                    max_norm=self.training_config.max_grad_norm,
                )

                self._optimizer_step()

                self.teacher.update_parameters(self.model)

                # tracking
                self.logger.update('lr', self.adamw_scheduler.get_last_lr()[0])
                self.logger.update('predictor_dropout', self._current_dropout)

                self.logger.update('ssl_loss_unscaled', acc_buffer.get_mean('ssl_loss_unscaled'))
                self.logger.update('ssl_loss', acc_buffer.get_mean('ssl_loss'))

                self.logger.update('legal_loss_unscaled', acc_buffer.get_mean('legal_loss_unscaled'))
                self.logger.update('legal_loss', acc_buffer.get_mean('legal_loss'))
                self.logger.update('legal_f1', binary_f1_from_stats(legal_f1_stats).item())

                self.logger.update('attacks_loss_unscaled', acc_buffer.get_mean('attacks_loss_unscaled'))
                self.logger.update('attacks_loss', acc_buffer.get_mean('attacks_loss'))
                self.logger.update(
                    'attacks_f1',
                    multiclass_f1_from_confusion_matrix(attacks_confusion).item()
                )

                self.logger.update(
                    'perceptive_legal_loss_unscaled',
                    acc_buffer.get_mean('perceptive_legal_loss_unscaled')
                )
                self.logger.update('perceptive_legal_loss', acc_buffer.get_mean('perceptive_legal_loss'))

                _perceptive_legal_f1 = binary_f1_from_stats(perceptive_legal_f1_stats).item()
                self.logger.update('perceptive_legal_f1', _perceptive_legal_f1)

                self._update_dropout(_perceptive_legal_f1)

                self.logger.update(
                    'perceptive_attacks_loss_unscaled',
                    acc_buffer.get_mean('perceptive_attacks_loss_unscaled')
                )
                self.logger.update('perceptive_attacks_loss', acc_buffer.get_mean('perceptive_attacks_loss'))
                self.logger.update(
                    'perceptive_attacks_f1',
                    multiclass_f1_from_confusion_matrix(perceptive_attacks_confusion).item()
                )
                # \ tracking

                acc_buffer.reset()
                legal_f1_stats.zero_()
                attacks_confusion.zero_()
                perceptive_legal_f1_stats.zero_()
                perceptive_attacks_confusion.zero_()
                pbar.update(1)

                self._optimizer_zero_grad(set_to_none=True)

                step += 1
                if step >= self.training_config.max_steps:
                    break

                if step >= 100 and step % 10 == 0:
                    pbar.set_description(
                        f'SSL Training | {
                            self.logger.log(step, exclude_if_contains=[
                                'unscaled',
                                'perceptive',
                                'attacks'
                            ])
                        }'
                    )

                if step % self.training_config.save_every == 0:
                    self.ckpt(
                        step=step,
                        training_state=True,
                        additional_modules={'predictor': self.predictor, 'teacher': self.teacher}
                    )


class LegalAttacksTrainer(Trainer):
    """Trainer for Legal and Attacks heads only, for ablation studies
    """
    def __init__(self, config, config_path):
        super().__init__(config, config_path)

        # type hints
        self.training_config: SSLTrainingConfig
        self.model_config: SSLModelConfig
        self.data_config: SSLDataConfig

        self.muon_optimizer: Muon
        self.adamw_optimizer: AdamW
        self.muon_scheduler: torch.optim.lr_scheduler.LambdaLR
        self.adamw_scheduler: torch.optim.lr_scheduler.LambdaLR
        # \ type hints

        self._build_optimizer()

    def _build_optimizer(self):
        muon_params = [
            p for n, p in self.model.named_parameters()
            if p.requires_grad and
            p.ndim == 2 and
            not any(nd in n for nd in ['head', 'embed'])
        ]
        muon_param_ids = {id(p) for p in muon_params}
        adamw_params = [
            p for n, p in self.model.named_parameters()
            if p.requires_grad and
            id(p) not in muon_param_ids
        ]

        self.muon_optimizer = Muon(
            muon_params,
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
            adjust_lr_fn='match_rms_adamw'
        )
        self.adamw_optimizer = AdamW(
            adamw_params,
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
            betas=(0.9, 0.98),
            eps=1e-7
        )

        lr_schedule_fn = wsd_schedule(
            warmup_steps=self.training_config.warmup_steps,
            max_steps=self.training_config.max_steps
        )
        self.muon_scheduler = LambdaLR(self.muon_optimizer, lr_schedule_fn)
        self.adamw_scheduler = LambdaLR(self.adamw_optimizer, lr_schedule_fn)

        del self.optimizer
        del self.scheduler

    def ckpt(self, step=None, training_state=True, additional_modules=None):
        if training_state:
            torch.save(self.muon_optimizer.state_dict(), os.path.join(self.logdir, 'muon_optimizer.pth'))
            torch.save(self.adamw_optimizer.state_dict(), os.path.join(self.logdir, 'adamw_optimizer.pth'))
            torch.save(self.muon_scheduler.state_dict(), os.path.join(self.logdir, 'muon_scheduler.pth'))
            torch.save(self.adamw_scheduler.state_dict(), os.path.join(self.logdir, 'adamw_scheduler.pth'))
        super().ckpt(step=step, training_state=False, additional_modules=additional_modules)

    def _optimizer_step(self):
        self.muon_optimizer.step()
        self.adamw_optimizer.step()

        self.muon_scheduler.step()
        self.adamw_scheduler.step()

    def _optimizer_zero_grad(self, set_to_none=True):
        self.muon_optimizer.zero_grad(set_to_none=set_to_none)
        self.adamw_optimizer.zero_grad(set_to_none=set_to_none)

    def _build_configs(self, config):
        self.training_config = SSLTrainingConfig(**config['training'])

        hidden_size = config['model']['hidden_size']
        encoder_config = CFEncoderConfig(hidden_size=hidden_size, **config['model'].pop('encoder'))

        self.model_config = SSLModelConfig(encoder_config=encoder_config, **config['model'])

        self.tracking_config = TrackingConfig(**config['tracking'])
        self.data_config = SSLDataConfig(**config['data'])

    def _build_model(self):
        self.model = SSLChessModel(self.model_config)

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
            collate_fn=SSLCollator(self.data_config.max_prediction_depth),
        )

    def run(self):
        self.model.train()
        self.model.to(self.device)

        self._optimizer_zero_grad(set_to_none=True)

        gradient_accumulation_steps = self.training_config.gradient_accumulation_steps

        step = 0
        total_steps = (len(self.train_dataloader)+gradient_accumulation_steps-1) // gradient_accumulation_steps
        total_steps = min(
            self.training_config.max_steps,
            total_steps
        )
        pbar = tqdm(total=total_steps, desc='No-SSL Training')

        acc_buffer = AccumulationBuffer(gradient_accumulation_steps, self.device)
        attacks_num_classes = self.model.attacks_head.output_dim
        legal_f1_stats = torch.zeros(3, device=self.device)
        attacks_confusion = torch.zeros((attacks_num_classes, attacks_num_classes), device=self.device)

        for partial_step, batch in enumerate(self.train_dataloader):
            tokens, _, legal_moves, attacks, _, _, _, _ = batch

            tokens = tokens.to(self.device, non_blocking=True)

            legal_moves = legal_moves.to(self.device, dtype=torch.float32, non_blocking=True)
            attacks = attacks.to(self.device, dtype=torch.long, non_blocking=True)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                _, logits, losses = self.model(
                    tokens,
                    target={'legal': legal_moves, 'attacks': attacks}
                )

                legal_logits = logits['legal']
                legal_loss = losses['legal']

                attacks_logits = logits['attacks']
                attacks_loss = losses['attacks']


                total_loss = legal_loss / gradient_accumulation_steps * self.training_config.legal_loss_weight
                total_loss += attacks_loss / gradient_accumulation_steps * self.training_config.attacks_loss_weight

            total_loss.backward()

            # metrics
            acc_buffer.update('legal_loss_unscaled', legal_loss.detach(), partial_step)
            acc_buffer.update(
                'legal_loss',
                legal_loss.detach() * self.training_config.legal_loss_weight,
                partial_step
            )
            legal_f1_stats += binary_f1_stats(legal_logits.detach(), legal_moves)

            acc_buffer.update('attacks_loss_unscaled', attacks_loss.detach(), partial_step)
            acc_buffer.update(
                'attacks_loss',
                attacks_loss.detach() * self.training_config.attacks_loss_weight,
                partial_step
            )
            attacks_confusion += multiclass_confusion_matrix(attacks_logits.detach(), attacks)
            # \ metrics

            if (partial_step + 1) % gradient_accumulation_steps == 0 or partial_step == len(self.train_dataloader):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.training_config.max_grad_norm,
                )

                self._optimizer_step()

                # tracking
                self.logger.update('lr', self.adamw_scheduler.get_last_lr()[0])

                self.logger.update('legal_loss_unscaled', acc_buffer.get_mean('legal_loss_unscaled'))
                self.logger.update('legal_loss', acc_buffer.get_mean('legal_loss'))
                self.logger.update('legal_f1', binary_f1_from_stats(legal_f1_stats).item())

                self.logger.update('attacks_loss_unscaled', acc_buffer.get_mean('attacks_loss_unscaled'))
                self.logger.update('attacks_loss', acc_buffer.get_mean('attacks_loss'))
                self.logger.update(
                    'attacks_f1',
                    multiclass_f1_from_confusion_matrix(attacks_confusion).item()
                )
                # \ tracking

                acc_buffer.reset()
                legal_f1_stats.zero_()
                attacks_confusion.zero_()
                pbar.update(1)

                self._optimizer_zero_grad(set_to_none=True)

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
                    )


class PVTuner(Trainer):
    """Fine-tunes policy/value heads of a pre-trained model. The encoder and embedding are
    frozen.
    """
    def __init__(self, config, config_path):
        super().__init__(config, config_path)
        # type hints
        self.training_config: PVTuningConfig
        self.model_config: SSLModelConfig
        self.data_config: Lc0DataConfig

        self.muon_optimizer: Muon | None
        self.adamw_optimizer: AdamW
        self.muon_scheduler: torch.optim.lr_scheduler.LambdaLR | None
        self.adamw_scheduler: torch.optim.lr_scheduler.LambdaLR
        # \ type hints

        self._build_optimizer()

    def _build_optimizer(self):
        encoder_lr_factor = self.training_config.encoder_lr_factor
        lr = self.training_config.learning_rate

        if encoder_lr_factor == 0.0:
            for param in self.model.encoder.parameters():
                param.requires_grad = False
            for param in self.model.embedding.parameters():
                param.requires_grad = False
            self.muon_optimizer = None
            self.muon_scheduler = None
            adamw_param_groups = [
                {'params': [p for p in self.model.parameters() if p.requires_grad], 'lr': lr},
            ]
        else:
            muon_params = [
                p for n, p in self.model.encoder.named_parameters()
                if p.requires_grad and p.ndim == 2
            ]
            muon_ids = {id(p) for p in muon_params}

            embedding_params = [
                p for p in self.model.embedding.parameters() if p.requires_grad
            ]
            embedding_ids = {id(p) for p in embedding_params}

            encoder_residual_params = [
                p for p in self.model.encoder.parameters()
                if p.requires_grad and id(p) not in muon_ids
            ]
            encoder_residual_ids = {id(p) for p in encoder_residual_params}

            head_params = [
                p for p in self.model.parameters()
                if p.requires_grad
                and id(p) not in muon_ids | embedding_ids | encoder_residual_ids
            ]

            self.muon_optimizer = Muon(
                muon_params,
                lr=lr * encoder_lr_factor,
                weight_decay=self.training_config.weight_decay,
                adjust_lr_fn='match_rms_adamw',
            )
            self.muon_scheduler = LambdaLR(
                self.muon_optimizer,
                wsd_schedule(
                    warmup_steps=self.training_config.warmup_steps,
                    max_steps=self.training_config.max_steps,
                )
            )
            adamw_param_groups = [
                {'params': embedding_params + encoder_residual_params, 'lr': lr * encoder_lr_factor},
                {'params': head_params, 'lr': lr},
            ]

        self.adamw_optimizer = AdamW(
            adamw_param_groups,
            weight_decay=self.training_config.weight_decay,
            betas=(0.9, 0.98),
            eps=1e-7,
        )
        lr_schedule_fn = wsd_schedule(
            warmup_steps=self.training_config.warmup_steps,
            max_steps=self.training_config.max_steps,
        )
        self.adamw_scheduler = LambdaLR(self.adamw_optimizer, lr_schedule_fn)

        del self.optimizer
        del self.scheduler

    def _build_configs(self, config):
        self.training_config = PVTuningConfig(**config['training'])

        hidden_size = config['model']['hidden_size']
        encoder_config = CFEncoderConfig(hidden_size=hidden_size, **config['model'].pop('encoder'))
        self.model_config = SSLModelConfig(encoder_config=encoder_config, **config['model'])

        self.tracking_config = TrackingConfig(**config['tracking'])
        self.data_config = Lc0DataConfig(**config['data'])

    def _build_model(self):
        ssl_model = SSLChessModel(self.model_config)
        state_dict = torch.load(
            self.training_config.ssl_model_path,
            map_location='cpu',
            weights_only=True,
        )
        ssl_model.load_state_dict(state_dict)
        self.model = ssl_to_pv_model(ssl_model)
        del ssl_model

    def ckpt(self, step=None, training_state=True, additional_modules=None):
        if training_state:
            if self.muon_optimizer is not None:
                torch.save(
                    self.muon_optimizer.state_dict(),
                    os.path.join(self.logdir, 'muon_optimizer.pth')
                )
                torch.save(
                    self.muon_scheduler.state_dict(),
                    os.path.join(self.logdir, 'muon_scheduler.pth')
                )
            torch.save(
                self.adamw_optimizer.state_dict(),
                os.path.join(self.logdir, 'adamw_optimizer.pth')
            )
            torch.save(
                self.adamw_scheduler.state_dict(),
                os.path.join(self.logdir, 'adamw_scheduler.pth')
            )
        super().ckpt(step=step, training_state=False, additional_modules=additional_modules)

    def _optimizer_step(self):
        if self.muon_optimizer is not None:
            self.muon_optimizer.step()
            self.muon_scheduler.step()
        self.adamw_optimizer.step()
        self.adamw_scheduler.step()

    def _optimizer_zero_grad(self, set_to_none=True):
        if self.muon_optimizer is not None:
            self.muon_optimizer.zero_grad(set_to_none=set_to_none)
        self.adamw_optimizer.zero_grad(set_to_none=set_to_none)

    def _load_datasets(self):
        dataset = Lc0PositionDataset(
            directory=self.data_config.directory,
            encoding=self.data_config.encoding,
        )
        print(f'{dataset.__class__.__name__} size: {len(dataset):,}')

        self.train_dataloader = DataLoader(
            dataset,
            batch_size=self.training_config.batch_size,
            shuffle=True,
            num_workers=self.training_config.num_workers,
            pin_memory=True,
        )

    def run(self):
        self.model.train()
        self.model.to(self.device)

        self._optimizer_zero_grad(set_to_none=True)

        gradient_accumulation_steps = self.training_config.gradient_accumulation_steps

        step = 0
        total_steps = (len(self.train_dataloader)+gradient_accumulation_steps-1) // gradient_accumulation_steps
        total_steps = min(
            self.training_config.max_steps,
            total_steps
        )
        pbar = tqdm(total=total_steps, desc='PV Tuning')

        acc_buffer = AccumulationBuffer(gradient_accumulation_steps, self.device)

        for partial_step, batch in enumerate(self.train_dataloader):
            tokens, target_dict = batch

            tokens = tokens.to(self.device, non_blocking=True)
            target = {
                k: v.to(self.device, non_blocking=True)
                for k, v in target_dict.items()
            }

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                _, policy_out, value_out = self.model(tokens, target)

                policy_loss = policy_out.loss
                value_loss = value_out.loss

                total_loss = policy_loss / gradient_accumulation_steps * \
                    self.training_config.policy_loss_weight
                if value_loss is not None:
                    total_loss += value_loss / gradient_accumulation_steps * \
                        self.training_config.value_loss_weight

            total_loss.backward()

            # metrics
            acc_buffer.update('policy_loss', policy_loss.detach(), partial_step)
            acc_buffer.update('value_loss', value_loss.detach(), partial_step)

            accuracy = (
                policy_out.logits.detach().argmax(dim=-1) == target['policy'].argmax(dim=-1)
            ).float().mean()
            acc_buffer.update('accuracy', accuracy, partial_step)

            top3_accuracy = (
                policy_out.logits.detach().topk(3, dim=-1).indices ==
                target['policy'].argmax(dim=-1, keepdim=True)
            ).any(dim=-1).float().mean()
            acc_buffer.update('top3_accuracy', top3_accuracy, partial_step)
            # \ metrics

            if (partial_step + 1) % gradient_accumulation_steps == 0 or partial_step == len(self.train_dataloader):
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    max_norm=self.training_config.max_grad_norm,
                )

                self._optimizer_step()

                # tracking
                self.logger.update('lr', self.adamw_scheduler.get_last_lr()[-1])
                self.logger.update('policy_loss', acc_buffer.get_mean('policy_loss'))
                self.logger.update('value_loss', acc_buffer.get_mean('value_loss', ignore_zeros=True))
                self.logger.update('accuracy', acc_buffer.get_mean('accuracy'))
                self.logger.update('top3_accuracy', acc_buffer.get_mean('top3_accuracy'))
                # \ tracking

                acc_buffer.reset()
                pbar.update(1)

                self._optimizer_zero_grad(set_to_none=True)

                step += 1
                if step >= self.training_config.max_steps:
                    break

                if step >= 100 and step % 10 == 0:
                    pbar.set_description(
                        f'PV Tuning | {self.logger.log(step)}'
                    )

                if step % self.training_config.save_every == 0:
                    self.ckpt(step=step, training_state=True)
