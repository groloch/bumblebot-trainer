import copy
import os

import mlflow
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import CFEncoderConfig, SSLModelConfig, TrackingConfig
from ..data import T91GamesDataset
from ..modeling import PositionClassificationHeads, SSLChessModel
from ..tracking import MetricLogger
from ..tracking.metrics import (
    accuracy_from_stats,
    accuracy_stats,
    multiclass_confusion_matrix,
    multiclass_f1_from_confusion_matrix,
)
from .utils import init_logdir, init_run, wsd_schedule, Trainer


class ClassificationTrainer(Trainer):
    """Trainer that trains expert routing heads on top of a pre-trained model backbone.
    Currently adapted to two experts groups:
    - activity: tactical or strategical position classification
    - evaluation: winning or drawish or losing position
    """
    def __init__(self, config, config_path):
        raw_config = copy.deepcopy(config)
        self.training_config = config['training']
        self.data_config = config['data']
        self.model_config_dict = config['model']
        self.pretrained_config = config['pretrained']
        self.tracking_config = TrackingConfig(**config['tracking'])

        self.logdir = init_logdir(self.training_config['logdir'], config_path)
        print(f'Logging to {self.logdir}')
        init_run(self.training_config['seed'], self.tracking_config, raw_config)

        self._build_models()
        self._load_datasets()
        self._build_optimizer()
        self.logger = MetricLogger(self.tracking_config.use_mlflow)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _build_models(self):
        model_config = copy.deepcopy(self.model_config_dict)
        hidden_size = model_config['hidden_size']
        encoder_config = CFEncoderConfig(
            hidden_size=hidden_size,
            **model_config.pop('encoder'),
        )
        model_config = SSLModelConfig(
            encoder_config=encoder_config,
            **model_config,
        )
        self.base_model = SSLChessModel(model_config)

        for param in self.base_model.parameters():
            param.requires_grad_(False)

        checkpoint_path = self.pretrained_config['checkpoint']
        if not os.path.isabs(checkpoint_path):
            checkpoint_path = os.path.join(self.pretrained_config['logdir'], checkpoint_path)
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f'Base checkpoint not found: {checkpoint_path}')

        state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        self.base_model.load_state_dict(state, strict=True)
        self.base_model.requires_grad_(False)
        if hasattr(self.base_model.encoder, 'layer_dropout'):
            self.base_model.encoder.layer_dropout = False

        self.heads = PositionClassificationHeads(hidden_size)

    def _load_datasets(self):
        dataset = T91GamesDataset(
            min_moves=self.data_config['min_moves'],
            encoding=self.data_config['encoding'],
            tag='all',
        )
        self.train_dataloader = DataLoader(
            dataset,
            batch_size=self.training_config['batch_size'],
            shuffle=True,
            num_workers=self.training_config['num_workers'],
            pin_memory=True,
        )

    def _build_optimizer(self):
        self.optimizer = AdamW(
            self.heads.parameters(),
            lr=self.training_config['learning_rate'],
            weight_decay=self.training_config['weight_decay'],
        )
        self.scheduler = LambdaLR(
            self.optimizer,
            wsd_schedule(
                warmup_steps=self.training_config['warmup_steps'],
                max_steps=self.training_config['max_steps'],
            ),
        )

    def ckpt(self, step=None):
        name = self.training_config['name'] if step is None else \
            f"{self.training_config['name']}_step{step}"
        torch.save(self.heads.state_dict(), os.path.join(self.logdir, f'{name}.pth'))

    def wrapup(self):
        if self.tracking_config.use_mlflow:
            mlflow.end_run()
        self.ckpt()

    def run(self):
        self.base_model.to(self.device).eval()
        self.heads.to(self.device).train()

        self.optimizer.zero_grad(set_to_none=True)
        gradient_accumulation_steps = self.training_config['gradient_accumulation_steps']
        max_steps = self.training_config['max_steps']
        pbar = tqdm(total=max_steps, desc='Classification Head Training')
        activity_stats = torch.zeros(2, device=self.device)
        evaluation_stats = torch.zeros(2, device=self.device)
        activity_confusion = torch.zeros((2, 2), device=self.device)
        evaluation_confusion = torch.zeros((3, 3), device=self.device)
        step = 0

        for partial_step, batch in enumerate(self.train_dataloader):
            tokens, targets, extra = batch
            tokens = tokens.to(self.device, non_blocking=True)
            targets = {key: value.to(self.device, non_blocking=True) for key, value in targets.items()}
            extra = {key: value.to(self.device, non_blocking=True) for key, value in extra.items()}
            activity_target = (
                targets['policy'].amax(dim=-1) > self.data_config.get('tactical_threshold', 0.6)
            ).long()
            evaluation_target = targets['value'].argmax(dim=-1)

            with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
                embeddings, _ = self.base_model.embed(tokens, **extra)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits, losses = self.heads(
                    embeddings,
                    activity_target=activity_target,
                    evaluation_target=evaluation_target,
                )
                activity_loss = losses['activity']
                evaluation_loss = losses['evaluation']
                total_loss = (
                    activity_loss * self.training_config.get('activity_loss_weight', 1.0)
                    + evaluation_loss * self.training_config.get('evaluation_loss_weight', 1.0)
                ) / gradient_accumulation_steps

            total_loss.backward()
            activity_logits = logits['activity'].detach()
            evaluation_logits = logits['evaluation'].detach()
            activity_stats += accuracy_stats(activity_logits, activity_target)
            evaluation_stats += accuracy_stats(evaluation_logits, evaluation_target)
            activity_confusion += multiclass_confusion_matrix(activity_logits, activity_target)
            evaluation_confusion += multiclass_confusion_matrix(
                evaluation_logits,
                evaluation_target
            )

            if (partial_step + 1) % gradient_accumulation_steps != 0:
                continue

            torch.nn.utils.clip_grad_norm_(
                self.heads.parameters(),
                max_norm=self.training_config['max_grad_norm'],
            )
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

            self.logger.update('lr', self.scheduler.get_last_lr()[0])
            self.logger.update('activity_loss', activity_loss.detach().item())
            self.logger.update('evaluation_loss', evaluation_loss.detach().item())
            self.logger.update('activity_accuracy', accuracy_from_stats(activity_stats).item())
            self.logger.update('evaluation_accuracy', accuracy_from_stats(evaluation_stats).item())
            self.logger.update(
                'activity_f1',
                multiclass_f1_from_confusion_matrix(activity_confusion).item()
            )
            self.logger.update(
                'evaluation_f1',
                multiclass_f1_from_confusion_matrix(evaluation_confusion).item()
            )
            activity_stats.zero_()
            evaluation_stats.zero_()
            activity_confusion.zero_()
            evaluation_confusion.zero_()
            step += 1
            pbar.update(1)

            if step >= 100 and step % 10 == 0:
                pbar.set_description(
                    f'Classification Head Training | {self.logger.log(step)}'
                )

            if step % self.training_config['save_every'] == 0:
                self.ckpt(step=step)

            if step >= max_steps:
                break
