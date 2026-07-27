import torch


@torch.no_grad()
def binary_f1_stats(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    predictions = logits > 0
    targets = targets > 0.5

    tp = (predictions & targets).sum(dtype=torch.float32)
    fp = (predictions & ~targets).sum(dtype=torch.float32)
    fn = (~predictions & targets).sum(dtype=torch.float32)
    return torch.stack([tp, fp, fn])


@torch.no_grad()
def binary_f1_from_stats(stats: torch.Tensor) -> torch.Tensor:
    tp, fp, fn = stats.unbind()
    denom = 2.0 * tp + fp + fn
    return torch.where(
        denom > 0,
        (2.0 * tp) / denom,
        torch.ones((), device=stats.device, dtype=torch.float32)
    )


def binary_f1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Standard f1 score
    """
    return binary_f1_from_stats(binary_f1_stats(logits, targets)).item()


@torch.no_grad()
def multiclass_confusion_matrix(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    predictions = torch.argmax(logits, dim=-1).reshape(-1)
    targets = targets.reshape(-1)
    num_classes = logits.shape[-1]

    flat_indices = targets * num_classes + predictions
    return torch.bincount(
        flat_indices,
        minlength=num_classes * num_classes
    ).reshape(num_classes, num_classes).to(dtype=torch.float32)


@torch.no_grad()
def multiclass_f1_from_confusion_matrix(confusion_matrix: torch.Tensor) -> torch.Tensor:
    tp = confusion_matrix.diag()
    fp = confusion_matrix.sum(dim=0) - tp
    fn = confusion_matrix.sum(dim=1) - tp

    denom = 2.0 * tp + fp + fn
    f1_scores = torch.where(
        denom > 0,
        (2.0 * tp) / denom,
        torch.ones_like(denom)
    )
    return f1_scores.mean()


def multiclass_f1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Macro-averaged f1 score
    """
    return multiclass_f1_from_confusion_matrix(multiclass_confusion_matrix(logits, targets)).item()

def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Standard accuracy
    """
    with torch.no_grad():
        predictions = torch.argmax(logits, dim=-1)
        correct = (predictions == targets).sum().float()
        total = targets.numel()
        accuracy = correct / total
    return accuracy.item()
