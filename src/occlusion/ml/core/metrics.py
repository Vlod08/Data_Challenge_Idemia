import torch


def weighted_error(pred: torch.Tensor, target: torch.Tensor, offset: float = 1.0 / 30.0) -> float:
    w = offset + target
    return ((w * (pred - target) ** 2).sum() / w.sum().clamp_min(1e-12)).item()


def challenge_score(preds: torch.Tensor, targets: torch.Tensor,
                    genders: torch.Tensor, offset: float = 1.0 / 30.0) -> dict:
    """Score = (Err_g0 + Err_g1) / 2 + |Err_g0 - Err_g1| (symmetric in F/M)."""
    preds, targets, genders = preds.flatten(), targets.flatten(), genders.flatten()
    errs = {}
    for g in (0.0, 1.0):
        mask = genders == g
        errs[g] = weighted_error(
            preds[mask], targets[mask], offset) if mask.any() else 0.0
    score = (errs[0.0] + errs[1.0]) / 2 + abs(errs[0.0] - errs[1.0])
    return {"score": score, "err_g0": errs[0.0], "err_g1": errs[1.0]}
