import torch


def binary_f1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Standard f1 score
    """
    with torch.no_grad():
        predictions = logits > 0
        targets = targets > 0.5
        tp = (predictions & targets).sum().float()
        fp = (predictions & ~targets).sum().float()
        fn = (~predictions & targets).sum().float()

        denom = 2.0 * tp + fp + fn
        if denom == 0:
            return 1.0
        f1 = (2.0 * tp) / denom
    return f1.item()

def multiclass_f1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Macro-averaged f1 score
    """
    with torch.no_grad():
        predictions = torch.argmax(logits, dim=-1)
        num_classes = logits.shape[-1]
        f1_scores = []
        for c in range(num_classes):
            tp = ((predictions == c) & (targets == c)).sum().float()
            fp = ((predictions == c) & (targets != c)).sum().float()
            fn = ((predictions != c) & (targets == c)).sum().float()
            denom = 2.0 * tp + fp + fn
            f1_scores.append((2.0 * tp / denom).item() if denom > 0 else 1.0)
    return sum(f1_scores) / len(f1_scores)

def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Standard accuracy
    """
    with torch.no_grad():
        predictions = torch.argmax(logits, dim=-1)
        correct = (predictions == targets).sum().float()
        total = targets.numel()
        accuracy = correct / total
    return accuracy.item()
