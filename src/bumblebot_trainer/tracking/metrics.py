import torch


def legal_f1(logits: torch.Tensor, targets: torch.Tensor) -> float:
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
