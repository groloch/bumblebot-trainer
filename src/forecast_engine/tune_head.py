import math
import os
from dataclasses import asdict

import numpy as np
import torch
import chess
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import mlflow

from .data.position_datasets import CombinedPositionDataset, LichessEvalDataset, PuzzleDataset
from .data.game_datasets import LC0GamesDataset
from .modeling.model import ChessModel
from .tracking.metric_logger import MetricLogger
from .utils import model_parameters
from .config.training_config import TuningConfig
from .config.modeling_configs import EncoderConfig, ForecastConfig, ModelConfig
from .config.tracking_config import TrackingConfig
from .train import init, wrapup, build_model_config, init_logdir


def tune(config: dict, config_path: str):
    training_config = TuningConfig(**config['training'])
    model_config = build_model_config(config['model'])
    tracking_config = TrackingConfig(**config['tracking'])

    init(training_config.seed, tracking_config)
    logdir = init_logdir(training_config.logdir, config_path)
    print(f'Logging to {logdir}')

    model = ChessModel(model_config)
    state_dict = torch.load(
        os.path.join(training_config.pretrained_logdir, 'forecast_model.pth'),
        map_location='cpu',
        weights_only=True
    )
    model.load_state_dict(state_dict)

    datasets = []
    if training_config.tune_value:
        datasets.append(LichessEvalDataset(training_config.forecast_depth, split='tune_value'))
    if training_config.tune_policy:
        pass
    if training_config.tune_forecast:
        pass

    dataset = CombinedPositionDataset(datasets)
    print('Data splits sizes:')
    for dataset_ in dataset.datasets:
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
        model.value_head.parameters(),
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

    tune_run(
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


def tune_run(
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

    optimizer.zero_grad(set_to_none=True)

    forecast_depth = model.forecast_head.forecast_depth

    step = 0
    total_steps = min(
        max_steps,
        (len(dataloader)+gradient_accumulation_steps-1) // gradient_accumulation_steps
    )

    loss_buffer = torch.zeros(
        gradient_accumulation_steps,
        device=device,
        dtype=torch.bfloat16
    )
    value_loss_buffer = torch.zeros(
        gradient_accumulation_steps,
        device=device,
        dtype=torch.bfloat16
    )

    pbar = tqdm(total=total_steps, desc='Training')

    for partial_step, batch in enumerate(dataloader, start=1):

        tokens, target_dict = batch

        tokens = tokens.to(device)
        target = {
            k: v.to(device) for k, v in target_dict.items()
        }
        target['value'] = target['value'].to(dtype=torch.bfloat16)

        with torch.no_grad():
            embeddings = model.embed(tokens)
        value_out = model.value_head(embeddings.cls_embedding, target['value'])

        total_loss = value_out.loss

        loss_buffer[partial_step % gradient_accumulation_steps] = total_loss.detach()
        value_loss_buffer[partial_step % gradient_accumulation_steps] = value_out.loss.detach()

        total_loss = total_loss / gradient_accumulation_steps
        total_loss.backward()


        if (partial_step+1) % gradient_accumulation_steps == 0 or (partial_step+1) == len(dataloader):
            torch.nn.utils.clip_grad_norm_(model.value_head.parameters(), max_norm=max_grad_norm)

            optimizer.step()
            scheduler.step()

            logger.update('train_loss', loss_buffer.mean().item())
            logger.update('value_loss', value_loss_buffer[value_loss_buffer != 0].mean().item())
            logger.update('lr', scheduler.get_last_lr()[0])

            loss_buffer.zero_()
            value_loss_buffer.zero_()

            pbar.update(1)

            optimizer.zero_grad(set_to_none=True)

            step += 1
            if step > max_steps:
                break

            if step >= 100 and step % 10 == 0:
                pbar.set_description(
                    f'Training | {logger.log(step, exclude_if_contains='forecast')}'
                )
            
            if (step+1) % save_every == 0:
                torch.save(model.state_dict(), os.path.join(logdir, f'forecast_model_{step}.pth'))
