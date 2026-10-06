import os

import numpy as np
import torch
import torch.nn as nn
import mlflow
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import EvaluationConfig
from ..data.game_datasets import LichessPuzzlesDataset
from .bayes_elo import bayes_elo


class PuzzleEvaluator:
    def __init__(
        self,
        model: nn.Module,
        config: EvaluationConfig,
        encoding: str,
        device: torch.device,
        use_mlflow: bool,
        logdir: str,
    ):
        self.model = model
        self.config = config
        self.device = device
        self.use_mlflow = use_mlflow
        self.logdir = logdir

        self.dataset = LichessPuzzlesDataset(
            encoding,
            num_puzzles=config.num_puzzles,
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=device.type == 'cuda',
            drop_last=False,
        )

        self.puzzle_lens = self.dataset.puzzle_lens
        self.num_puzzles = self.dataset.num_puzzles

    def _policy_logits(self, tokens: torch.Tensor) -> torch.Tensor:
        squares_embeddings, _ = self.model.embed(tokens)
        return self.model.policy_head(squares_embeddings, None).logits

    @torch.no_grad()
    def run(self, step: int) -> dict[str, float]:
        was_training = self.model.training
        self.model.eval()

        n = self.num_puzzles
        correct_per_puzzle = np.zeros(n, dtype=np.float64)
        total_per_puzzle = np.zeros(n, dtype=np.int64)
        rating_per_puzzle = np.zeros(n, dtype=np.float64)

        n_positions = 0
        n_correct = 0

        use_autocast = self.device.type == 'cuda'

        pbar = tqdm(self.dataloader, desc='Solving puzzles', leave=False)
        for tokens, move_ids, puzzle_ids, ratings in pbar:
            tokens = tokens.to(self.device, non_blocking=True)
            move_ids = move_ids.to(self.device, non_blocking=True)

            puzzle_ids_np = puzzle_ids.numpy()
            ratings_np = ratings.numpy()

            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=use_autocast,
            ):
                logits = self._policy_logits(tokens)

            predictions = logits.argmax(dim=-1)
            correct = (predictions == move_ids).cpu().numpy()

            n_positions += correct.size
            n_correct += int(correct.sum())

            correct_per_puzzle += np.bincount(
                puzzle_ids_np, weights=correct, minlength=n
            )
            total_per_puzzle += np.bincount(puzzle_ids_np, minlength=n)
            rating_per_puzzle[puzzle_ids_np] = ratings_np

            pbar.set_postfix(accuracy=f'{n_correct / max(n_positions, 1):.3f}')

        self.model.train(was_training)

        # A puzzle is solved only if every one of its solution moves was found
        fully_evaluated = total_per_puzzle == self.puzzle_lens
        solved = fully_evaluated & (correct_per_puzzle >= total_per_puzzle)

        evaluated = total_per_puzzle > 0
        ratings = rating_per_puzzle[evaluated]
        outcomes = solved[evaluated].astype(np.float64)

        elo, elo_std = bayes_elo(
            ratings,
            outcomes,
            prior_mean=self.config.prior_elo_mean,
            prior_std=self.config.prior_elo_std,
            elo_min=self.config.elo_min,
            elo_max=self.config.elo_max,
            elo_step=self.config.elo_step,
        )

        metrics = {
            'puzzle/elo': elo,
            'puzzle/elo_std': elo_std,
            'puzzle/accuracy': n_correct / max(n_positions, 1),
            'puzzle/solved': float(solved.sum()),
            'puzzle/solved_rate': float(solved.sum()) / max(int(evaluated.sum()), 1),
        }

        if self.use_mlflow:
            mlflow.log_metrics(metrics, step=step)

        return metrics
