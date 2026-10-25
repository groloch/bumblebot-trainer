import copy
import os

import mlflow
import torch
from torch.optim import AdamW, Muon
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import (
    CFEncoderConfig,
    EvaluationConfig,
    SSLModelConfig,
    TrackingConfig,
    TrainingConfig
)
from ..data import (
    CombinedPositionDataset,
    LichessPuzzlesDataset,
    PositionCollator,
    T91GamesDataset,
)
from ..eval import PuzzleEvaluator
from ..modeling import SSLChessModel
from ..modeling.utils import load_model_state_dict
from ..tracking import AccumulationBuffer, MetricLogger
from ..tracking.metrics import (
    accuracy_from_stats,
    accuracy_stats,
    topk_accuracy_stats,
)
from .utils import init_logdir, init_run, model_parameters, wsd_schedule, move_optimizer_state, Trainer


class PVTrainer(Trainer):
    """Trainer for policy-value-only training.
    """
    def __init__(self, config, config_path):
        print('Loading run configuration')
        raw_config = copy.deepcopy(config)
        self._build_configs(config)

        self.logdir = init_logdir(self.training_config['logdir'], config_path)
        print(f'Logging to {self.logdir}')
        init_run(self.training_config['seed'], self.tracking_config, raw_config)

        print('Building model')
        self._build_model()
        total_params, trainable_params = model_parameters(self.model)
        print(
            f'Model Parameters: {total_params / 1e6:.2f}M total, '
            f'{trainable_params / 1e6:.2f}M trainable'
        )

        print('Loading dataset')
        self._load_datasets()

        print('Building optimizer and scheduler')
        self._build_optimizer()

        self.logger = MetricLogger(self.tracking_config.use_mlflow)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _build_configs(self, config):
        self.training_config = config['training']
        self.data_config = config['data']
        self.model_config_dict = config['model']
        self.pretrained_config = config.get('pretrained')
        self.tracking_config = TrackingConfig(**config['tracking'])
        self.eval_config = EvaluationConfig(**config.get('evaluation', {}))

    def _build_model(self):
        model_config = copy.deepcopy(self.model_config_dict)
        hidden_size = model_config['hidden_size']
        encoder_config = CFEncoderConfig(
            hidden_size=hidden_size,
            **model_config.pop('encoder'),
        )
        self.model_config = SSLModelConfig(
            encoder_config=encoder_config,
            **model_config,
        )

        checkpoint = (
            self.pretrained_config.get('checkpoint')
            if self.pretrained_config is not None
            else None
        )
        label_smoothing = 0.2 if checkpoint is None else 0
        self.model = SSLChessModel(self.model_config, label_smoothing=label_smoothing)

        if checkpoint is not None:
            checkpoint_path = checkpoint
            if not os.path.isabs(checkpoint_path):
                checkpoint_path = os.path.join(
                    self.pretrained_config.get('logdir', ''), checkpoint_path
                )
            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(f'Pretrained checkpoint not found: {checkpoint_path}')
            state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            if isinstance(state, dict) and 'state_dict' in state:
                state = state['state_dict']
            load_model_state_dict(self.model, state)
            print(f'Loaded pretrained weights from {checkpoint_path}')

        if hasattr(self.model.encoder, 'layer_dropout') and checkpoint is not None:
            self.model.encoder.layer_dropout = False

    def _load_datasets(self):
        tag = self.data_config.get('tag', 'all').lower()
        t91_dataset = T91GamesDataset(
            min_moves=self.data_config['min_moves'],
            encoding=self.data_config['encoding'],
            tag=tag,
        )
        if tag == 'tactical':
            puzzle_dataset = LichessPuzzlesDataset(
                encoding=self.data_config['encoding'],
                split='train',
            )
            dataset = CombinedPositionDataset([t91_dataset, puzzle_dataset])
        else:
            dataset = t91_dataset

        self.train_dataloader = DataLoader(
            dataset,
            batch_size=self.training_config['batch_size'],
            shuffle=True,
            num_workers=self.training_config['num_workers'],
            pin_memory=True,
            collate_fn=PositionCollator(),
        )

    def _build_optimizer(self):
        muon_params = [
            parameter
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
            and parameter.ndim == 2
            and not any(excluded in name for excluded in ('head', 'embed'))
        ]
        muon_param_ids = {id(parameter) for parameter in muon_params}
        adamw_params = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad and id(parameter) not in muon_param_ids
        ]

        self.muon_optimizer = Muon(
            muon_params,
            lr=self.training_config['learning_rate'],
            weight_decay=self.training_config['weight_decay'],
            adjust_lr_fn='match_rms_adamw',
        )
        self.adamw_optimizer = AdamW(
            adamw_params,
            lr=self.training_config['learning_rate'],
            weight_decay=self.training_config['weight_decay'],
            betas=(0.9, 0.98),
            eps=1e-7,
        )

        schedule = wsd_schedule(
            warmup_steps=self.training_config['warmup_steps'],
            max_steps=self.training_config['max_steps'],
        )
        self.muon_scheduler = LambdaLR(self.muon_optimizer, schedule)
        self.adamw_scheduler = LambdaLR(self.adamw_optimizer, schedule)

    def _optimizer_step(self):
        self.muon_optimizer.step()
        self.adamw_optimizer.step()
        self.muon_scheduler.step()
        self.adamw_scheduler.step()

    def _optimizer_zero_grad(self, set_to_none=True):
        self.muon_optimizer.zero_grad(set_to_none=set_to_none)
        self.adamw_optimizer.zero_grad(set_to_none=set_to_none)

    def ckpt(self, step=None, training_state=True):
        if training_state:
            for name, obj in (
                ('muon_optimizer', self.muon_optimizer),
                ('adamw_optimizer', self.adamw_optimizer),
                ('muon_scheduler', self.muon_scheduler),
                ('adamw_scheduler', self.adamw_scheduler),
            ):
                torch.save(obj.state_dict(), os.path.join(self.logdir, f'{name}.pth'))

        model_name = f"{self.training_config['name']}.pth" if step is None else \
            f"{self.training_config['name']}_step{step}.pth"
        torch.save(self.model.state_dict(), os.path.join(self.logdir, model_name))

    def wrapup(self):
        if self.tracking_config.use_mlflow:
            mlflow.end_run()
        self.ckpt(training_state=False)

    def run(self):
        self.model.train()
        self.model.to(self.device)

        move_optimizer_state(self.muon_optimizer, self.device)
        move_optimizer_state(self.adamw_optimizer, self.device)

        print('Loading puzzle evaluation dataset')
        self.evaluator = PuzzleEvaluator(
            model=self.model,
            config=self.eval_config,
            encoding=self.data_config['encoding'],
            device=self.device,
            use_mlflow=self.tracking_config.use_mlflow,
            logdir=self.logdir,
        )

        self._optimizer_zero_grad(set_to_none=True)
        gradient_accumulation_steps = self.training_config['gradient_accumulation_steps']
        max_steps = self.training_config['max_steps']
        pbar = tqdm(total=max_steps, desc='PV Fine-tuning')

        acc_buffer = AccumulationBuffer(gradient_accumulation_steps, self.device)
        policy_stats = torch.zeros(2, device=self.device)
        policy_top3_stats = torch.zeros(2, device=self.device)

        step = 0
        for partial_step, batch in enumerate(self.train_dataloader):
            tokens, targets, extra = batch
            tokens = tokens.to(self.device, non_blocking=True)
            targets = {key: value.to(self.device, non_blocking=True) for key, value in targets.items()}
            extra = {key: value.to(self.device, non_blocking=True) for key, value in extra.items()}

            extra['progress'] = step / max_steps

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                _, logits, losses = self.model(tokens, target=targets, **extra)
                policy_loss = losses['policy']
                value_loss = losses['value']
                total_loss = (
                    policy_loss * self.training_config['policy_loss_weight']
                    + value_loss * self.training_config['value_loss_weight']
                ) / gradient_accumulation_steps

            total_loss.backward()

            acc_buffer.update('policy_loss_unscaled', policy_loss.detach(), partial_step)
            acc_buffer.update(
                'policy_loss',
                policy_loss.detach() * self.training_config['policy_loss_weight'],
                partial_step,
            )
            acc_buffer.update('value_loss_unscaled', value_loss.detach(), partial_step)
            acc_buffer.update(
                'value_loss',
                value_loss.detach() * self.training_config['value_loss_weight'],
                partial_step,
            )
            policy_stats += accuracy_stats(logits['policy'].detach(), targets['policy'])
            policy_top3_stats += topk_accuracy_stats(logits['policy'].detach(), targets['policy'])

            if (partial_step + 1) % gradient_accumulation_steps != 0:
                continue

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.training_config['max_grad_norm'],
            )
            self._optimizer_step()

            self.logger.update('lr', self.adamw_scheduler.get_last_lr()[0])
            self.logger.update('policy_loss_unscaled', acc_buffer.get_mean('policy_loss_unscaled'))
            self.logger.update('policy_loss', acc_buffer.get_mean('policy_loss'))
            self.logger.update('policy_accuracy', accuracy_from_stats(policy_stats).item())
            self.logger.update('policy_top3_accuracy', accuracy_from_stats(policy_top3_stats).item())
            self.logger.update('value_loss_unscaled', acc_buffer.get_mean('value_loss_unscaled'))
            self.logger.update('value_loss', acc_buffer.get_mean('value_loss'))

            acc_buffer.reset()
            policy_stats.zero_()
            policy_top3_stats.zero_()
            self._optimizer_zero_grad(set_to_none=True)
            step += 1
            pbar.update(1)

            if step % self.eval_config.every == 0:
                eval_metrics = self.evaluator.run(step)
                pbar.write(
                    f'[step {step}] puzzle Elo: {eval_metrics["puzzle/elo"]:.0f} '
                    f'± {eval_metrics["puzzle/elo_std"]:.0f} | '
                    f'solved: {eval_metrics["puzzle/solved_rate"]:.2%} | '
                    f'move accuracy: {eval_metrics["puzzle/move_accuracy"]:.2%}'
                )

            if step >= max_steps:
                break

            if step >= 100 and step % 10 == 0:
                pbar.set_description(
                    f'PV Training | {self.logger.log(step, exclude_if_contains=["unscaled"])}'
                )

            if step % self.training_config['save_every'] == 0:
                self.ckpt(step=step, training_state=True)
