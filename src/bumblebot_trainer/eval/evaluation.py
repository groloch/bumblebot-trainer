from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import mlflow
from torch.utils.data import default_collate
from tqdm import tqdm

from ..config import EvaluationConfig
from ..data.game_datasets import LichessPuzzlesDataset
from ..data.position_datasets import LichessEvalDataset
from ..utils import expected_result_to_cp, wdl_to_expected_result
from .bayes_elo import bayes_elo
from .utils import (
    autocast,
    get_model_embeddings,
    make_dataloader,
    preserve_model_mode,
)


GAME_PHASES = ('opening', 'middlegame', 'endgame')


def _pad_move_ids(move_id_lists: list[torch.Tensor]) -> torch.Tensor:
    max_num_moves = max(move_ids.numel() for move_ids in move_id_lists)
    padded = torch.full((len(move_id_lists), max_num_moves), -1, dtype=torch.long)
    for row, move_ids in enumerate(move_id_lists):
        padded[row, :move_ids.numel()] = move_ids
    return padded


def _collate_puzzle_batch(batch):
    samples = default_collate([item[:5] for item in batch])
    acceptable_move_ids = _pad_move_ids([item[5] for item in batch])
    confidence_move_ids = _pad_move_ids([item[6] for item in batch])
    tag_masks = default_collate([item[7] for item in batch])
    return (*samples, acceptable_move_ids, confidence_move_ids, tag_masks)


def _mean_or_zero(total: float, count: int) -> float:
    return total / count if count else 0.0


class PuzzleEvaluator:
    """Evaluation on puzzles from lichess.

    Evaluates elo, move accuracy, solved rate (solving a puzzle requires several best moves to be
    found, this is therefore different than move accuracy), confidence (best move predicted prob).

    Also measures those metrics by-theme (see LichessPuzzlesDataset for the selected themes).
    """
    def __init__(
        self,
        model: nn.Module,
        config: EvaluationConfig,
        encoding: str,
        device: torch.device,
        use_mlflow: bool,
        logdir: str = '',
    ):
        self.model = model
        self.config = config
        self.device = device
        self.use_mlflow = use_mlflow
        self.logdir = logdir

        self.dataset = LichessPuzzlesDataset(encoding, split='valid')
        self.dataloader = make_dataloader(
            self.dataset, config, device, collate_fn=_collate_puzzle_batch
        )

        self.puzzle_lens = self.dataset.puzzle_lens
        self.num_puzzles = self.dataset.num_puzzles
        self.tag_names = self.dataset.TAGS

    def _policy_logits(self, tokens: torch.Tensor, **kwargs) -> torch.Tensor:
        squares, _ = get_model_embeddings(self.model, tokens, kwargs)
        return self.model.policy_head(squares, None).logits

    @torch.no_grad()
    @preserve_model_mode
    def run(self, step: int) -> dict[str, float]:
        n = self.num_puzzles
        correct_per_puzzle = np.zeros(n, dtype=np.float64)
        total_per_puzzle = np.zeros(n, dtype=np.int64)
        rating_per_puzzle = np.zeros(n, dtype=np.float64)
        tags_per_puzzle = np.zeros((n, len(self.tag_names)), dtype=np.bool_)

        confidence_sum = 0.0
        confidence_count = 0
        confidence_sum_per_tag = np.zeros(len(self.tag_names), dtype=np.float64)
        confidence_count_per_tag = np.zeros(len(self.tag_names), dtype=np.int64)
        confidence_sum_per_elo: dict[int, float] = defaultdict(float)
        confidence_count_per_elo: dict[int, int] = defaultdict(int)

        n_positions = 0
        n_correct = 0

        pbar = tqdm(self.dataloader, desc='Solving puzzles', leave=False)
        for (
            tokens,
            _move_ids,
            puzzle_ids,
            ratings,
            extra,
            acceptable_move_ids,
            confidence_move_ids,
            tag_masks,
        ) in pbar:
            tokens = tokens.to(self.device, non_blocking=True)
            acceptable_move_ids = acceptable_move_ids.to(self.device, non_blocking=True)
            confidence_move_ids = confidence_move_ids.to(self.device, non_blocking=True)
            extra = {k: v.to(self.device, non_blocking=True) for k, v in extra.items()}

            puzzle_ids_np = puzzle_ids.numpy()
            ratings_np = ratings.numpy()
            tag_masks_np = tag_masks.numpy()

            with autocast(self.device):
                logits = self._policy_logits(tokens, **extra)

            probabilities = torch.softmax(logits.float(), dim=-1)
            predictions = logits.argmax(dim=-1)
            correct = (
                (predictions[:, None] == acceptable_move_ids)
                .logical_and(acceptable_move_ids >= 0)
                .any(dim=1)
            )
            confidence_ids_valid = confidence_move_ids >= 0
            confidence = (
                probabilities.gather(1, confidence_move_ids.clamp_min(0))
                * confidence_ids_valid
            ).sum(dim=1)

            correct_np = correct.cpu().numpy()
            confidence_np = confidence.cpu().numpy()
            n_positions += correct_np.size
            n_correct += int(correct_np.sum())

            correct_per_puzzle += np.bincount(
                puzzle_ids_np, weights=correct_np, minlength=n
            )
            total_per_puzzle += np.bincount(puzzle_ids_np, minlength=n)
            rating_per_puzzle[puzzle_ids_np] = ratings_np
            tags_per_puzzle[puzzle_ids_np] = tag_masks_np

            confidence_sum += float(confidence_np.sum())
            confidence_count += confidence_np.size
            confidence_sum_per_tag += (tag_masks_np * confidence_np[:, None]).sum(axis=0)
            confidence_count_per_tag += tag_masks_np.sum(axis=0)

            elo_buckets = (np.floor(ratings_np / 500).astype(np.int64) * 500)
            for bucket in np.unique(elo_buckets):
                in_bucket = elo_buckets == bucket
                confidence_sum_per_elo[int(bucket)] += float(confidence_np[in_bucket].sum())
                confidence_count_per_elo[int(bucket)] += int(in_bucket.sum())

            pbar.set_postfix(accuracy=f'{n_correct / max(n_positions, 1):.3f}')

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

        def grouped_elo(mask):
            return bayes_elo(
                rating_per_puzzle[mask],
                solved[mask].astype(np.float64),
                prior_mean=self.config.prior_elo_mean,
                prior_std=self.config.prior_elo_std,
                elo_min=self.config.elo_min,
                elo_max=self.config.elo_max,
                elo_step=self.config.elo_step,
            )

        solved_count = int(solved[evaluated].sum())
        evaluated_count = int(evaluated.sum())
        metrics = {
            'puzzle/elo': float(elo),
            'puzzle/elo_std': float(elo_std),
            'puzzle/move_accuracy': n_correct / max(n_positions, 1),
            'puzzle/solved_rate': solved_count / max(evaluated_count, 1),
            'puzzle/confidence': _mean_or_zero(confidence_sum, confidence_count),
        }

        for tag_idx, tag in enumerate(self.tag_names):
            tagged = evaluated & tags_per_puzzle[:, tag_idx]
            tag_solved = int((solved & tagged).sum())
            tag_count = int(tagged.sum())
            prefix = f'puzzle/tag/{tag}'
            metrics[f'{prefix}/solved_rate'] = tag_solved / max(tag_count, 1)
            tag_elo, tag_elo_std = grouped_elo(tagged)
            metrics[f'{prefix}/elo'] = float(tag_elo)
            metrics[f'{prefix}/elo_std'] = float(tag_elo_std)
            metrics[f'{prefix}/confidence'] = _mean_or_zero(
                confidence_sum_per_tag[tag_idx],
                int(confidence_count_per_tag[tag_idx]),
            )

        elo_buckets = (np.floor(rating_per_puzzle / 500).astype(np.int64) * 500)
        for bucket in sorted(set(elo_buckets[evaluated].tolist())):
            in_bucket = evaluated & (elo_buckets == bucket)
            bucket_solved = int((solved & in_bucket).sum())
            bucket_count = int(in_bucket.sum())
            prefix = f'puzzle/elo_range/{bucket}-{bucket + 499}'
            metrics[f'{prefix}/solved_rate'] = bucket_solved / max(bucket_count, 1)
            metrics[f'{prefix}/confidence'] = _mean_or_zero(
                confidence_sum_per_elo[int(bucket)],
                confidence_count_per_elo[int(bucket)],
            )

        if self.use_mlflow:
            mlflow.log_metrics(metrics, step=step)

        return metrics


class StockfishEvaluator:
    """Evaluation on stockfish annotated positions (LichessEval datasets).
    Evaluates top-1 and top-3 accuracy, kl divergence, and confidence on the top-1 move.

    Also measures those metrics per-game-phase
    """
    TOP_K = 3

    def __init__(
        self,
        model: nn.Module,
        config: EvaluationConfig,
        encoding: str,
        device: torch.device,
        use_mlflow: bool,
        logdir: str = '',
    ):
        self.model = model
        self.device = device
        self.use_mlflow = use_mlflow
        self.logdir = logdir

        self.dataset = LichessEvalDataset(encoding, split='valid')
        self.dataloader = make_dataloader(self.dataset, config, device)

    @torch.no_grad()
    @preserve_model_mode
    def run(self, step: int) -> dict[str, float]:
        stats = defaultdict(lambda: {
            'count': 0,
            'correct': 0,
            'topk_correct': 0,
            'kl_sum': 0.0,
            'target_confidence_sum': 0.0,
        })
        wdl_abs_error_sum = 0.0
        wdl_cp_loss_sum = 0.0
        qvalue_abs_error_sum = 0.0
        qvalue_cp_loss_sum = 0.0
        qvalue_count = 0
        value_from_logits = isinstance(
            getattr(self.model.value_head, 'loss_fn', None),
            nn.CrossEntropyLoss,
        )

        def accumulate_policy_stats(name, mask, correct, topk_correct, kl, target_confidence):
            count = int(mask.sum().item())
            if count == 0:
                return
            group = stats[name]
            group['count'] += count
            group['correct'] += int((correct & mask).sum().item())
            group['topk_correct'] += int((topk_correct & mask).sum().item())
            group['kl_sum'] += float(kl[mask].sum().item())
            group['target_confidence_sum'] += float(target_confidence[mask].sum().item())

        pbar = tqdm(self.dataloader, desc='Evaluating Stockfish positions', leave=False)
        for tokens, targets, extra in pbar:
            tokens = tokens.to(self.device, non_blocking=True)
            targets = {k: v.to(self.device, non_blocking=True) for k, v in targets.items()}
            extra = {k: v.to(self.device, non_blocking=True) for k, v in extra.items()}

            with autocast(self.device):
                squares, value_embedding = get_model_embeddings(self.model, tokens, extra)
                policy_logits = self.model.policy_head(squares, None).logits
                value_output = self.model.value_head(value_embedding, None).logits
                qvalue_output = self.model.qvalue_head(value_embedding, None).logits

            policy_targets = targets['policy'].float()
            policy_targets = policy_targets.clamp_min(0.0)
            policy_targets = policy_targets / policy_targets.sum(dim=-1, keepdim=True).clamp_min(1e-12)

            target_moves = policy_targets.argmax(dim=-1)
            predictions = policy_logits.argmax(dim=-1)
            correct = predictions == target_moves
            k = min(self.TOP_K, policy_logits.shape[-1])
            topk_correct = (
                policy_logits.topk(k, dim=-1).indices == target_moves[:, None]
            ).any(dim=-1)
            log_probabilities = F.log_softmax(policy_logits.float(), dim=-1)
            kl = F.kl_div(
                log_probabilities,
                policy_targets,
                reduction='none',
            ).sum(dim=-1)
            target_confidence = policy_targets.max(dim=-1).values

            accumulate_policy_stats(
                'all',
                torch.ones_like(correct, dtype=torch.bool),
                correct,
                topk_correct,
                kl,
                target_confidence,
            )

            phase_ids = targets['game_phase'].long().reshape(-1)
            for phase_idx, phase in enumerate(GAME_PHASES):
                accumulate_policy_stats(
                    f'phase/{phase}',
                    phase_ids == phase_idx,
                    correct,
                    topk_correct,
                    kl,
                    target_confidence,
                )

            predicted_wdl_result = wdl_to_expected_result(
                value_output,
                from_logits=value_from_logits,
            )
            predicted_qvalue = qvalue_output.float().reshape(-1).clamp(0.0, 1.0)
            target_qvalue = targets['qvalue'].float().reshape(-1).clamp(0.0, 1.0)
            stockfish_cp = targets['stockfish_cp'].float().reshape(-1)

            wdl_cp = expected_result_to_cp(predicted_wdl_result)
            qvalue_cp = expected_result_to_cp(predicted_qvalue)
            wdl_abs_error_sum += float(
                (predicted_wdl_result - target_qvalue).abs().sum().item()
            )
            wdl_cp_loss_sum += float((wdl_cp - stockfish_cp).abs().sum().item())
            qvalue_abs_error_sum += float(
                (predicted_qvalue - target_qvalue).abs().sum().item()
            )
            qvalue_cp_loss_sum += float((qvalue_cp - stockfish_cp).abs().sum().item())
            qvalue_count += int(target_qvalue.numel())

        metrics: dict[str, float] = {}
        for name, group in stats.items():
            count = group['count']
            prefix = f'stockfish/policy/{name}' if name != 'all' else 'stockfish/policy'
            metrics[f'{prefix}/positions'] = float(count)
            metrics[f'{prefix}/accuracy'] = _mean_or_zero(group['correct'], count)
            metrics[f'{prefix}/top{self.TOP_K}_accuracy'] = _mean_or_zero(
                group['topk_correct'], count
            )
            metrics[f'{prefix}/kl_divergence'] = _mean_or_zero(group['kl_sum'], count)
            metrics[f'{prefix}/target_confidence'] = _mean_or_zero(
                group['target_confidence_sum'], count
            )

        metrics['stockfish/wdl/expected_result_mae'] = _mean_or_zero(
            wdl_abs_error_sum, qvalue_count
        )
        metrics['stockfish/wdl/centipawn_loss'] = _mean_or_zero(
            wdl_cp_loss_sum, qvalue_count
        )
        metrics['stockfish/qvalue/expected_result_mae'] = _mean_or_zero(
            qvalue_abs_error_sum, qvalue_count
        )
        metrics['stockfish/qvalue/centipawn_loss'] = _mean_or_zero(
            qvalue_cp_loss_sum, qvalue_count
        )

        if self.use_mlflow:
            mlflow.log_metrics(metrics, step=step)

        return metrics


class EvaluationSuite:
    """Puzzle and Stockfish evaluation datasets, reusable across training steps."""

    def __init__(
        self,
        model: nn.Module,
        config: EvaluationConfig,
        encoding: str,
        device: torch.device,
        use_mlflow: bool = False,
        logdir: str = '',
    ):
        self.puzzle_evaluator = PuzzleEvaluator(
            model, config, encoding, device, use_mlflow, logdir
        )
        self.stockfish_evaluator = StockfishEvaluator(
            model, config, encoding, device, use_mlflow, logdir
        )

    def run(self, step: int) -> dict[str, dict[str, float]]:
        return {
            'puzzle': self.puzzle_evaluator.run(step),
            'stockfish': self.stockfish_evaluator.run(step),
        }


def run_evaluation_suite(
    model: nn.Module,
    config: EvaluationConfig,
    encoding: str,
    device: torch.device,
    use_mlflow: bool = False,
    logdir: str = '',
    step: int = 0,
) -> dict[str, dict[str, float]]:
    """Run puzzle and Stockfish-position evaluation on an existing model."""
    return EvaluationSuite(model, config, encoding, device, use_mlflow, logdir).run(step)


def run_evaluation_suite_from_logdir(
    logdir: str,
    checkpoint_path: str | None = None,
    *,
    use_mlflow: bool = False,
    step: int = 0,
) -> dict[str, dict[str, float]]:
    """Load a trained checkpoint from a run directory and evaluate it."""
    from .stockfish_evaluation import _load_model_from_logdir

    run_dir = Path(logdir).expanduser().resolve()
    if not run_dir.is_dir():
        raise NotADirectoryError(f'Model logdir not found: {run_dir}')

    model, encoding, run_config, checkpoint, device = _load_model_from_logdir(
        run_dir,
        checkpoint_path,
    )
    eval_values = run_config.get('evaluation', {})
    eval_config = EvaluationConfig(**{
        key: value
        for key, value in eval_values.items()
        if key in EvaluationConfig.__dataclass_fields__
    })

    print(f'Running evaluation suite for {checkpoint}')
    metrics = run_evaluation_suite(
        model=model,
        config=eval_config,
        encoding=encoding,
        device=device,
        use_mlflow=use_mlflow,
        logdir=str(run_dir),
        step=step,
    )
    for evaluator_name, evaluator_metrics in metrics.items():
        print(f'[{evaluator_name}]')
        for name, value in sorted(evaluator_metrics.items()):
            print(f'{name}: {value:.6f}')
    return metrics
