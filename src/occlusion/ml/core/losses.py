import torch
import torch.nn as nn


def _weighted_mse(pred, target, offset):
    w = offset + target
    return (w * (pred - target) ** 2).sum() / w.sum().clamp_min(1e-12)


class WeightedMSELoss(nn.Module):
    """Weighted MSE matching the challenge metric: w = 1/30 + GT (gender ignored)."""

    def __init__(self, offset: float = 1.0 / 30.0):
        super().__init__()
        self.offset = offset

    def forward(self, pred, target, gender=None):
        return _weighted_mse(pred, target, self.offset)


class GenderBalancedWeightedMSELoss(nn.Module):
    """Per-gender weighted MSE averaged over genders. Targets (Err_F + Err_M)/2
    and implicitly shrinks the |Err_F - Err_M| disparity term of the Score."""

    def __init__(self, offset: float = 1.0 / 30.0):
        super().__init__()
        self.offset = offset

    def forward(self, pred, target, gender):
        errs = [_weighted_mse(pred[gender == g], target[gender == g], self.offset)
                for g in (0., 1.) if (gender == g).any()]
        return torch.stack(errs).mean()


def build_loss(name: str = "balanced", offset: float = 1.0 / 30.0) -> nn.Module:
    if name == "balanced":
        return GenderBalancedWeightedMSELoss(offset)
    if name == "wmse":
        return WeightedMSELoss(offset)
    raise ValueError(f"unknown loss: {name}")
