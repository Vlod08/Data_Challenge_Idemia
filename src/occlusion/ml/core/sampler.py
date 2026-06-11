"""Weighted sampling modes (the loss already handles the gender mean, so the
sampler is mainly about exposure to the rare high-occlusion regime):

  - gender      : inverse gender frequency (balance the gender marginal).
  - occ         : (1/30 + occ)^p, oversample high occlusion regardless of gender.
  - gender_occ  : gender balance x occ; note the gender term favours the
                  (already better-fit) minority gender, so it is mildly misaligned.
  - cell        : inverse frequency of the (gender x occ-bin) cell, raised to p.
                  Equalises the gender x occlusion grid -> directly boosts the
                  rarest cell (minority gender's high-occlusion tail).

`occ_power` (p) controls strength; keep modest (~0.5-1.5) to avoid double
counting with the w = 1/30 + GT loss weight.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler

SAMPLER_MODES = ("gender", "occ", "gender_occ", "cell")
# Tail-aware occlusion bins for the `cell` mode (the extreme tail is its own cell).
CELL_BINS = [0.0, 0.05, 0.1, 0.2, 0.3, 1.01]


def _inv_freq(series: pd.Series) -> np.ndarray:
    return 1.0 / np.clip(series.map(series.value_counts(normalize=True)).to_numpy(),
                         1e-12, None)


def compute_sample_weights(catalog: pd.DataFrame, mode: str, occ_power: float = 0.5,
                           occ_offset: float = 1.0 / 30.0,
                           occ_col: str = "FaceOcclusion",
                           gender_col: str = "gender") -> np.ndarray:
    occ = catalog[occ_col].to_numpy()
    if mode == "gender":
        w = _inv_freq(catalog[gender_col])
    elif mode == "occ":
        w = (occ_offset + occ) ** occ_power
    elif mode == "gender_occ":
        w = _inv_freq(catalog[gender_col]) * (occ_offset + occ) ** occ_power
    elif mode == "cell":
        occ_bin = pd.cut(catalog[occ_col], CELL_BINS, right=False).astype(str)
        cell = catalog[gender_col].astype(str) + "_" + occ_bin
        w = _inv_freq(cell) ** occ_power
    else:
        raise ValueError(f"unknown sampler mode: {mode}")
    return w / w.mean()


def build_weighted_sampler(catalog: pd.DataFrame, mode: str, occ_power: float = 0.5,
                           num_samples: int | None = None,
                           seed: int | None = None) -> WeightedRandomSampler:
    weights = compute_sample_weights(catalog, mode=mode, occ_power=occ_power)
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=num_samples or len(catalog),
        replacement=True, generator=generator)
