import torch
import torch.nn as nn


class WeightedMSELoss(nn.Module):
    """MSE pondérée alignée sur la métrique du challenge : w = 1/30 + GT."""

    def __init__(self, offset: float = 1.0 / 30.0):
        super().__init__()
        self.offset = offset

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        w = self.offset + target
        return (w * (pred - target) ** 2).sum() / w.sum().clamp_min(1e-12)
