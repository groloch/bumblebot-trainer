import math
import os
import sys
import random
import shutil

import numpy as np
import torch
import chess
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import mlflow


from .data.position_datasets import CombinedPositionDataset, LichessEvalDataset, PuzzleDataset, TablebaseDataset
from .data.game_datasets import LC0GamesDataset
from .modeling.model import ChessModel
from .tracking.metric_logger import MetricLogger, AccumulationBuffer
from .utils import model_parameters, ForecastVocabulary
from .config.training_config import TrainingConfig
from .config.modeling_configs import EncoderConfig, ForecastConfig, ModelConfig
from .config.tracking_config import TrackingConfig
from .config.data_config import DataConfig


def init(seed: int, tracking_config: TrackingConfig):
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

def wrapup(tracking_config: TrackingConfig, model: ChessModel, logdir: str):
    if tracking_config.use_mlflow:
        mlflow.end_run()
    torch.save(model.state_dict(), os.path.join(logdir, 'forecast_model.pth'))

def build_model_config(config: dict) -> ModelConfig:
    hidden_size = config['hidden_size']
    encoder_config = EncoderConfig(hidden_size=hidden_size, **config.pop('encoder'))
    forecast_config = ForecastConfig(hidden_size=hidden_size, **config.pop('forecast'))
    return ModelConfig(
        encoder_config=encoder_config,
        forecast_config=forecast_config,
        **config
    )

def init_logdir(logdir: str, config_path: str) -> str:
    i = 0
    proposed_logdir = os.path.join(logdir, f'run_{i}')
    while os.path.exists(proposed_logdir):
        i += 1
        proposed_logdir = os.path.join(logdir, f'run_{i}')
    os.makedirs(proposed_logdir)
    shutil.copy2(config_path, proposed_logdir)
    return proposed_logdir

def train(config: dict, config_path: str):
    training_config = TrainingConfig(**config['training'])
    model_config = build_model_config(config['model'])
    tracking_config = TrackingConfig(**config['tracking'])
    data_config = DataConfig(**config['data'])

    init(training_config.seed, tracking_config)
    logdir = init_logdir(training_config.logdir, config_path)
    print(f'Logging to {logdir}')

    model = ChessModel(model_config)

    datasets = []
    if 'lichess-eval' in data_config.datasets:
        datasets.append(LichessEvalDataset(training_config.forecast_depth))
    if 'puzzles' in data_config.datasets:
        datasets.append(PuzzleDataset(training_config.forecast_depth))
    if 'lc0' in data_config.datasets:
        datasets.append(LC0GamesDataset(training_config.forecast_depth, min_moves=30))
    if 'tb' in data_config.datasets:
        datasets.append(
            TablebaseDataset(
                training_config.forecast_depth, data_config.tablebase_path, data_config.fens_path
            )
        )

    dataset = CombinedPositionDataset(datasets)
    print('Data splits sizes:')
    for dataset_ in datasets:
        print(f'{dataset_.__class__.__name__} size: {len(dataset_):,}')
    print(f'Total: {len(dataset):,}')

    train_dataloader = DataLoader(
        dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=training_config.num_workers,
        pin_memory=True
    )

    optimizer = AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay
    )

    def lr_lambda(current_step):
        if current_step < training_config.warmup_steps:
            return float(current_step) / float(max(1, training_config.warmup_steps))
        decay_start = training_config.max_steps * 0.9
        if current_step < decay_start:
            return 1.0
        progress = (current_step - decay_start) / (training_config.max_steps - decay_start)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    metric_logger = MetricLogger(tracking_config.use_mlflow)

    total_params, trainable_params = model_parameters(model)
    print(f'Model Parameters: {total_params/1e6:.2f}M total, {trainable_params/1e6:.2f}M trainable')

    train_run(
        model=model,
        dataloader=train_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=torch.device('cuda'),
        max_steps=training_config.max_steps,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        max_grad_norm=training_config.max_grad_norm,
        forecast_loss_weight=training_config.forecast_loss_weight,
        logger=metric_logger,
        save_every=training_config.save_every,
        logdir=logdir
    )

    wrapup(tracking_config, model, logdir)

def train_run(
    model: ChessModel,
    dataloader: DataLoader,
    optimizer: AdamW,
    scheduler,
    device: torch.device,
    max_steps: int,
    gradient_accumulation_steps: int,
    max_grad_norm: float,
    forecast_loss_weight: float,
    logger: MetricLogger,
    save_every: int,
    logdir: str
):
    model.train()
    model.to(device)
    model.to(torch.bfloat16)

    forecast_depth = model.forecast_head.forecast_depth

    optimizer.zero_grad(set_to_none=True)

    step = 0
    total_steps = min(
        max_steps,
        (len(dataloader)+gradient_accumulation_steps-1) // gradient_accumulation_steps
    )
    pbar = tqdm(total=total_steps, desc='Training')

    acc_buffer = AccumulationBuffer(gradient_accumulation_steps, device)

    for partial_step, batch in enumerate(dataloader, start=1):
        tokens, target_dict = batch

        tokens = tokens.to(device)
        target = {
            k: v.to(device, dtype=torch.bfloat16)
            if k not in ('forecast_mask', 'forecast') else v.to(device)
            for k, v in target_dict.items()
        }

        _, policy_out, value_out, forecast_out = model(
            tokens,
            None, # trajectory,
            None, # target['forecast_mask'],
            target
        )

        total_loss = policy_out.loss
        acc_buffer.update('policy_loss', policy_out.loss.detach(), partial_step)

        if value_out.loss is not None:
            total_loss += value_out.loss
            acc_buffer.update('value_loss', value_out.loss.detach(), partial_step)

        if forecast_loss_weight > 0 and forecast_out is not None:
            total_loss += forecast_out.loss * forecast_loss_weight
            total_loss += forecast_out.horizon_loss * forecast_loss_weight

            acc_buffer.update('forecast_loss', forecast_out.loss.detach(), partial_step)
            acc_buffer.update('horizon_loss', forecast_out.horizon_loss.detach(), partial_step)

            flat_loss_mask = target['loss_mask'].view(-1)
            if flat_loss_mask.any():
                forecast_acc = (
                    forecast_out.logits.reshape(-1, model.forecast_head.n_classes).argmax(-1)[flat_loss_mask]
                    == (target['forecast'].reshape(-1)[flat_loss_mask] % ForecastVocabulary.HORIZON_OFFSET)
                ).float().mean()
                acc_buffer.update('forecast_accuracy', forecast_acc.detach(), partial_step)

                horizon_acc = (
                    forecast_out.horizon_logits.reshape(-1, 3).argmax(-1)[flat_loss_mask]
                    == (target['forecast'].reshape(-1)[flat_loss_mask] // ForecastVocabulary.HORIZON_OFFSET)
                ).float().mean()
                acc_buffer.update('horizon_accuracy', horizon_acc.detach(), partial_step)

        acc_buffer.update('train_loss', total_loss.detach(), partial_step)

        accuracy = (policy_out.logits.argmax(dim=-1) == target['policy'].argmax(dim=-1)).float().mean()
        acc_buffer.update('accuracy', accuracy.detach(), partial_step)

        total_loss = total_loss / gradient_accumulation_steps
        total_loss.backward()

        if partial_step % gradient_accumulation_steps == 0 or partial_step == len(dataloader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

            optimizer.step()
            scheduler.step()

            logger.update('train_loss', acc_buffer.get_mean('train_loss'))
            logger.update('policy_loss', acc_buffer.get_mean('policy_loss'))
            if value_out.loss is not None:
                logger.update('value_loss', acc_buffer.get_mean('value_loss', ignore_zeros=True))
            if forecast_loss_weight > 0 and forecast_out is not None:
                logger.update('forecast_loss', acc_buffer.get_mean('forecast_loss'))
                logger.update('forecast_accuracy', acc_buffer.get_mean('forecast_accuracy', ignore_zeros=True))
                logger.update('horizon_loss', acc_buffer.get_mean('horizon_loss'))
                logger.update('horizon_accuracy', acc_buffer.get_mean('horizon_accuracy', ignore_zeros=True))
            logger.update('lr', scheduler.get_last_lr()[0])
            logger.update('accuracy', acc_buffer.get_mean('accuracy'))

            acc_buffer.reset()
            pbar.update(1)

            optimizer.zero_grad(set_to_none=True)

            step += 1
            if step >= max_steps:
                break

            if step >= 100 and step % 10 == 0:
                pbar.set_description(
                    f'Training | {logger.log(step, exclude_if_contains=['forecast', 'horizon'])}'
                )

            if step % save_every == 0:
                torch.save(model.state_dict(), os.path.join(logdir, f'forecast_model_{step}.pth'))
