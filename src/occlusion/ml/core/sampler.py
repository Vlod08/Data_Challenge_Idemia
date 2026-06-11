"""Weighted sampling to balance the batch, aligned with the metric:
  - gender balance: the Score averages Err_F and Err_M and penalises their gap,
    so each gender should be equally represented.
  - occlusion tail: high occlusions are rare (~3% above 0.3) but heavily weighted
    by w = 1/30 + GT, so we can mildly oversample them.

Note: the loss already weights by w. Keep `occ_power` modest (0 to ~0.5) to
avoid double counting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler


def compute_sample_weights(catalog: pd.DataFrame, gender_balance: bool = True,
                           occ_power: float = 0.0, occ_offset: float = 1.0 / 30.0,
                           occ_col: str = "FaceOcclusion",
                           gender_col: str = "gender") -> np.ndarray:
    w = np.ones(len(catalog), dtype=np.float64)
    if gender_balance:
        freq = catalog[gender_col].map(
            catalog[gender_col].value_counts(normalize=True)).to_numpy()
        w /= np.clip(freq, 1e-12, None)
    if occ_power > 0:
        w *= (occ_offset + catalog[occ_col].to_numpy()) ** occ_power
    return w / w.mean()


def build_weighted_sampler(catalog: pd.DataFrame, gender_balance: bool = True,
                           occ_power: float = 0.0, occ_offset: float = 1.0 / 30.0,
                           num_samples: int | None = None,
                           seed: int | None = None) -> WeightedRandomSampler:
    weights = compute_sample_weights(catalog, gender_balance=gender_balance,
                                     occ_power=occ_power, occ_offset=occ_offset)
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=num_samples or len(catalog),
        replacement=True, generator=generator)
