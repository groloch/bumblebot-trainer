import os
import copy
import itertools

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import AdamW, Muon
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
import mlflow

from .utils import (
    init_logdir,
    init_run,
    model_parameters,
    wsd_schedule,
    dropout_schedule,
    Trainer
)
from ..modeling import SSLChessModel, Predictor
from ..data import (
    LichessStandardIterableSSLDataset,
    LichessStandardGamesSSLDataset,
    LichessPuzzlesSSLDataset,
    LichessEvalSSLDataset,
    T91GamesSSLDataset,
    T91GamesIterableSSLDataset,
    MixedSSLDataset,
    MixedSSLIterableDataset,
    SSLCollator
)
from ..eval import EvaluationSuite
from ..tracking import AccumulationBuffer, MetricLogger
from ..tracking.metrics import (
    accuracy_from_stats,
    accuracy_stats,
    topk_accuracy_stats,
    mae_from_stats,
    mae_stats,
)
from ..config import (
    TrainingConfig,
    SSLModelConfig,
    CFEncoderConfig,
    SSLDataConfig,
    PredictorConfig,
    TrackingConfig,
    DropoutScheduleConfig,
    EvaluationConfig,
)


class SSLTrainer(Trainer):
    """Trainer for the SSL pipeline, inspired by V-JEPA 2-AC.
    The model and its predictor are supervised on the following training objectives:
    - policy
    - value (wdl)
    - self-supervised loss: whether the predictor can recreate the future latent, given
    a partially masked move sequence encoding, and the current latent. Since the move sequence
    is partially masked, the predictor needs the missing information over the move sequence to
    be available (encoded) in the current position latent; in other words the model needs to have
    an implicit representation of the possible evolutions of the board state: the model needs to
    calculate.
    - perceptive losses: whether the predicted future latent is usable as-is: it could be close in
    distance but be adversarial wrt the teacher's heads. The output of the teacher heads (policy,
    value) on the predicted latent are compared to the teacher outputs directly on the future
    position.
    """
    def __init__(self, config, config_path):
        # type hints
        self.teacher: torch.nn.Module
        self.predictor: torch.nn.Module

        self.training_config: TrainingConfig
        self.model_config: SSLModelConfig
        self.data_config: SSLDataConfig

        self.muon_optimizer: Muon
        self.adamw_optimizer: AdamW
        self.muon_scheduler: torch.optim.lr_scheduler.LambdaLR
        self.adamw_scheduler: torch.optim.lr_scheduler.LambdaLR
        # \ type hints

        print('Loading run configuration')
        raw_config = copy.deepcopy(config)
        self._build_configs(config)

        self.logdir = init_logdir(self.training_config.logdir, config_path)
        print(f'Logging to {self.logdir}')

        init_run(self.training_config.seed, self.tracking_config, raw_config)

        print('Building model')
        self._build_model()
        total_params, trainable_params = model_parameters(self.model)
        print(f'Model Parameters: {total_params/1e6:.2f}M total, {trainable_params/1e6:.2f}M trainable')

        print('Loading datasets')
        self._load_datasets()

        print('Building optimizer and scheduler')
        self._build_optimizer()

        self.logger = MetricLogger(self.tracking_config.use_mlflow)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    def ckpt(self, step=None, training_state=True, additional_modules=None):
        if training_state:
            torch.save(
                self.muon_optimizer.state_dict(),
                os.path.join(self.logdir, 'muon_optimizer.pth')
            )
            torch.save(
                self.adamw_optimizer.state_dict(),
                os.path.join(self.logdir, 'adamw_optimizer.pth')
            )
            torch.save(
                self.muon_scheduler.state_dict(),
                os.path.join(self.logdir, 'muon_scheduler.pth')
            )
            torch.save(
                self.adamw_scheduler.state_dict(),
                os.path.join(self.logdir, 'adamw_scheduler.pth')
            )

        model_name = f'{self.training_config.name}.pth' if step is None else \
            f'{self.training_config.name}_step{step}.pth'
        torch.save(self.model.state_dict(), os.path.join(self.logdir, model_name))

        if additional_modules is not None:
            for name, module in additional_modules.items():
                torch.save(module.state_dict(), os.path.join(self.logdir, f'{name}.pth'))

    def wrapup(self):
        if self.tracking_config.use_mlflow:
            mlflow.end_run()
        self.ckpt(training_state=False)

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
        self.training_config = TrainingConfig(dropout_schedule=dropout_schedule_cfg, **config['training'])

        hidden_size = config['model']['hidden_size']
        encoder_config = CFEncoderConfig(hidden_size=hidden_size, **config['model'].pop('encoder'))

        self.model_config = SSLModelConfig(encoder_config=encoder_config, **config['model'])

        self.predictor_config = PredictorConfig(hidden_size=self.model_config.hidden_size, **config['predictor'])
        self.tracking_config = TrackingConfig(**config['tracking'])
        self.data_config = SSLDataConfig(**config['data'])

        eval_config = config.pop('evaluation', {})
        self.eval_config = EvaluationConfig(**eval_config)

    def _build_model(self):
        self.model = SSLChessModel(self.model_config)
        self.teacher = AveragedModel(self.model, multi_avg_fn=get_ema_multi_avg_fn(self.training_config.ema_decay))
        self.predictor = Predictor(self.predictor_config)

        for p in self.teacher.parameters():
            p.requires_grad = False

    def _update_dropout(self, perceptive_policy_accuracy: float):
        ds = self.training_config.dropout_schedule
        self._steps_since_dropout_update += 1
        if (perceptive_policy_accuracy >= ds.policy_accuracy_threshold
                and self._steps_since_dropout_update >= ds.min_steps_between_updates):
            self._n_dropout_updates += 1
            self._steps_since_dropout_update = 0
            self._current_dropout = self._dropout_fn(self._n_dropout_updates)

    def _load_datasets(self):
        dataset_classes = {
            't91': (T91GamesSSLDataset, T91GamesIterableSSLDataset),
            'lichess': (LichessStandardGamesSSLDataset, LichessStandardIterableSSLDataset),
            'puzzles': (LichessPuzzlesSSLDataset, None),
            'lichess_eval': (LichessEvalSSLDataset, None),
        }
        datasets = []
        weights = []

        for name, weight in self.data_config.dataset_mix.items():
            if weight <= 0:
                continue
            if name not in dataset_classes:
                raise ValueError(
                    f'Unknown SSL dataset {name}; expected one of {tuple(dataset_classes)}'
                )

            dataset_idx = 1 if self.data_config.streaming else 0
            dataset_class = dataset_classes[name][dataset_idx]
            if dataset_class is None:
                raise ValueError(f'SSL dataset {name} does not support streaming')

            dataset_kwargs = {
                'encoding': self.data_config.encoding,
                'max_prediction_depth': self.data_config.max_prediction_depth,
            }
            if name in ('t91', 'lichess'):
                dataset_kwargs['min_moves'] = self.data_config.min_moves
            if name == 'lichess' and self.data_config.streaming:
                dataset_kwargs['min_elo'] = self.data_config.min_elo

            datasets.append(dataset_class(**dataset_kwargs))
            weights.append(weight)

        if not datasets:
            raise ValueError('dataset_mix must contain at least one dataset with positive weight')

        if self.data_config.streaming:
            dataset = MixedSSLIterableDataset(datasets, weights)
        else:
            dataset = MixedSSLDataset(datasets, weights) if len(datasets) > 1 else datasets[0]

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
        # self.model.compile(mode='max-autotune')

        self.predictor.train()
        self.predictor.to(self.device)
        # self.predictor.compile(mode='max-autotune')

        self.teacher.eval()
        self.teacher.to(self.device)
        # self.teacher.compile(mode='max-autotune')

        print('Loading puzzle and Stockfish evaluation datasets')
        self.evaluator = EvaluationSuite(
            model=self.model,
            config=self.eval_config,
            encoding=self.data_config.encoding,
            device=self.device,
            use_mlflow=self.tracking_config.use_mlflow,
            logdir=self.logdir,
        )

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
        total_steps = self.training_config.max_steps
        pbar = tqdm(total=total_steps, desc='SSL Training')

        acc_buffer = AccumulationBuffer(gradient_accumulation_steps, self.device)
        policy_stats = torch.zeros(2, device=self.device)
        policy_top3_stats = torch.zeros(2, device=self.device)
        perceptive_policy_stats = torch.zeros(2, device=self.device)
        perceptive_policy_top3_stats = torch.zeros(2, device=self.device)

        tokens: torch.Tensor
        tokens_: torch.Tensor
        targets: dict[str, torch.Tensor]
        moves: torch.Tensor
        moves_attention_mask: torch.Tensor

        extra: dict[str, torch.Tensor]
        extra_: dict[str, torch.Tensor]

        for partial_step, batch in enumerate(self.train_dataloader):
            tokens, tokens_, targets, moves, moves_attention_mask, extra, extra_ = batch

            tokens = tokens.to(self.device, non_blocking=True)
            tokens_ = tokens_.to(self.device, non_blocking=True)

            targets = {k: v.to(self.device, non_blocking=True) for k, v in targets.items()}
            target = targets
            valid_ssl_indices = targets['valid_ssl_indices']

            moves = moves.to(self.device, non_blocking=True)
            moves_attention_mask = moves_attention_mask.to(self.device, non_blocking=True)

            extra = {k: v.to(self.device, non_blocking=True) for k, v in extra.items()}
            extra_ = {k: v.to(self.device, non_blocking=True) for k, v in extra_.items()}

            extra['progress'] = step / self.training_config.max_steps

            policy_weight = self.training_config.policy_loss_weight
            value_weight = self.training_config.value_loss_weight
            ssl_weight = self.training_config.ssl_loss_weight
            perceptive_weight = self.training_config.perceptive_loss_weight

            compute_student_losses = any(
                w > 0 for w in (policy_weight, value_weight)
            )
            compute_perceptive_losses = compute_student_losses and perceptive_weight > 0

            # jepa-like forward pass
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                if valid_ssl_indices.numel() > 0:
                    with torch.no_grad():
                        future_embed, target_embed = self.teacher.module.embed(tokens_, **extra_)
                        future_policy_logits = self.teacher.module.policy_head(future_embed).logits
                        future_value = self.teacher.module.value_head(future_embed.mean(dim=1)).logits
                        if compute_perceptive_losses:
                            perceptive_target = {
                                'policy': F.softmax(future_policy_logits.float(), dim=-1),
                                'value': F.softmax(future_value.float(), dim=-1),
                            }
                        else:
                            perceptive_target = {
                                'policy': future_policy_logits,
                                'value': future_value,
                            }
                else:
                    target_embed = None

                student_embed, logits, losses = self.model(
                    tokens,
                    target=target,
                    compute_losses=compute_student_losses,
                    **extra
                )

                if valid_ssl_indices.numel() > 0:
                    ssl_student_embed = torch.index_select(student_embed, 0, valid_ssl_indices)
                    pred_raw, pred_norm = self.predictor(
                        ssl_student_embed,
                        moves,
                        moves_attention_mask,
                        self._current_dropout
                    )
                    ssl_loss = (1 - F.cosine_similarity(pred_norm, target_embed.detach(), dim=-1)).mean()

                    perceptive_logits, perceptive_losses = self.teacher.module.heads_out(
                        pred_raw,
                        target=perceptive_target,
                        compute_losses=compute_perceptive_losses
                    )
                    perceptive_policy_loss = perceptive_losses['policy']
                    perceptive_policy_logits = perceptive_logits['policy']
                    perceptive_value_loss = perceptive_losses['value']
                else:
                    ssl_loss = student_embed.sum() * 0.0
                    perceptive_policy_loss = student_embed.sum() * 0.0
                    perceptive_value_loss = student_embed.sum() * 0.0
                    perceptive_policy_logits = logits['policy'].new_empty(
                        (0, logits['policy'].shape[-1])
                    )
                    perceptive_target = {
                        'policy': perceptive_policy_logits.new_empty(perceptive_policy_logits.shape),
                    }

                policy_loss = losses['policy']
                value_loss = losses['value']

                total_loss = ssl_loss / gradient_accumulation_steps * ssl_weight

                if policy_weight:
                    total_loss += policy_loss / gradient_accumulation_steps * policy_weight
                if value_weight:
                    total_loss += value_loss / gradient_accumulation_steps * value_weight

                if perceptive_weight:
                    perceptive_scale = perceptive_weight / gradient_accumulation_steps
                    if policy_weight:
                        total_loss += perceptive_policy_loss * perceptive_scale * policy_weight
                    if value_weight:
                        total_loss += perceptive_value_loss * perceptive_scale * value_weight

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

            acc_buffer.update('policy_loss_unscaled', policy_loss.detach(), partial_step)
            acc_buffer.update(
                'policy_loss',
                policy_loss.detach() * self.training_config.policy_loss_weight,
                partial_step
            )
            policy_stats += accuracy_stats(logits['policy'].detach(), target['policy'])
            policy_top3_stats += topk_accuracy_stats(logits['policy'].detach(), target['policy'])

            acc_buffer.update('value_loss_unscaled', value_loss.detach(), partial_step)
            acc_buffer.update(
                'value_loss',
                value_loss.detach() * self.training_config.value_loss_weight,
                partial_step
            )

            acc_buffer.update(
                'perceptive_policy_loss_unscaled', perceptive_policy_loss.detach(),
                partial_step
            )
            acc_buffer.update(
                'perceptive_policy_loss', perceptive_policy_loss.detach() * \
                    self.training_config.policy_loss_weight * self.training_config.perceptive_loss_weight,
                partial_step
            )
            perceptive_policy_stats += accuracy_stats(
                perceptive_policy_logits.detach(),
                perceptive_target['policy']
            )
            perceptive_policy_top3_stats += topk_accuracy_stats(
                perceptive_policy_logits.detach(),
                perceptive_target['policy']
            )

            acc_buffer.update(
                'perceptive_value_loss_unscaled', perceptive_value_loss.detach(),
                partial_step
            )
            acc_buffer.update(
                'perceptive_value_loss', perceptive_value_loss.detach() * \
                    self.training_config.value_loss_weight * self.training_config.perceptive_loss_weight,
                partial_step
            )
            # \ metrics

            if (partial_step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.parameters()) + list(self.predictor.parameters()),
                    max_norm=self.training_config.max_grad_norm,
                )

                self._optimizer_step()

                self.teacher.update_parameters(self.model)

                torch.cuda.synchronize()

                # tracking
                self.logger.update('lr', self.adamw_scheduler.get_last_lr()[0])
                self.logger.update('predictor_dropout', self._current_dropout)

                self.logger.update('ssl_loss_unscaled', acc_buffer.get_mean('ssl_loss_unscaled'))
                self.logger.update('ssl_loss', acc_buffer.get_mean('ssl_loss'))

                self.logger.update('policy_loss_unscaled', acc_buffer.get_mean('policy_loss_unscaled'))
                self.logger.update('policy_loss', acc_buffer.get_mean('policy_loss'))
                self.logger.update('policy_accuracy', accuracy_from_stats(policy_stats).item())
                self.logger.update('policy_top3_accuracy', accuracy_from_stats(policy_top3_stats).item())

                self.logger.update('value_loss_unscaled', acc_buffer.get_mean('value_loss_unscaled'))
                self.logger.update('value_loss', acc_buffer.get_mean('value_loss'))

                self.logger.update(
                    'perceptive_policy_loss_unscaled',
                    acc_buffer.get_mean('perceptive_policy_loss_unscaled')
                )
                self.logger.update('perceptive_policy_loss', acc_buffer.get_mean('perceptive_policy_loss'))

                _perceptive_policy_accuracy = accuracy_from_stats(perceptive_policy_stats).item()
                self.logger.update('perceptive_policy_accuracy', _perceptive_policy_accuracy)
                self.logger.update(
                    'perceptive_policy_top3_accuracy',
                    accuracy_from_stats(perceptive_policy_top3_stats).item()
                )

                self._update_dropout(_perceptive_policy_accuracy)

                self.logger.update(
                    'perceptive_value_loss_unscaled',
                    acc_buffer.get_mean('perceptive_value_loss_unscaled')
                )
                self.logger.update('perceptive_value_loss', acc_buffer.get_mean('perceptive_value_loss'))
                # \ tracking

                acc_buffer.reset()
                policy_stats.zero_()
                policy_top3_stats.zero_()
                perceptive_policy_stats.zero_()
                perceptive_policy_top3_stats.zero_()
                pbar.update(1)

                self._optimizer_zero_grad(set_to_none=True)

                step += 1

                if step % self.eval_config.every == 0:
                    eval_metrics = self.evaluator.run(step)
                    puzzle_metrics = eval_metrics['puzzle']
                    stockfish_metrics = eval_metrics['stockfish']
                    pbar.write(
                        f'[step {step}] puzzle Elo: {puzzle_metrics["puzzle/elo"]:.0f} '
                        f'± {puzzle_metrics["puzzle/elo_std"]:.0f} | '
                        f'solved: {puzzle_metrics["puzzle/solved_rate"]:.2%} | '
                        f'move accuracy: {puzzle_metrics["puzzle/move_accuracy"]:.2%} | '
                        f'Stockfish policy accuracy: '
                        f'{stockfish_metrics["stockfish/policy/accuracy"]:.2%} | '
                        f'top-3: {stockfish_metrics["stockfish/policy/top3_accuracy"]:.2%} | '
                        f'WDL MAE: '
                        f'{stockfish_metrics["stockfish/wdl/expected_result_mae"]:.3f}'
                    )

                if step >= self.training_config.max_steps:
                    break

                if step >= 100 and step % 10 == 0:
                    pbar.set_description(
                        f'SSL Training | {
                            self.logger.log(step, exclude_if_contains=[
                                'unscaled',
                                'perceptive',
                                'loss'
                            ])
                        }'
                    )

                if step % self.training_config.save_every == 0:
                    self.ckpt(
                        step=step,
                        training_state=True,
                        additional_modules={'predictor': self.predictor, 'teacher': self.teacher}
                    )
