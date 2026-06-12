"""Predict P(gender == 1) for every test image with a trained classifier.

Reads the run's config.json + model_best.pth, runs inference over the test
catalog, and writes artifacts/models/<run>/gender_pred.csv (filename, p_gender1).
Consumed by the post-hoc level-down step.

  uv run python scripts/predict_gender.py --run gender_b_lora
  uv run python scripts/predict_gender.py --run gender_b_lora --tta
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from occlusion import utils
from occlusion.config import PATHS
from occlusion.ml.core.gender import predict_gender
from occlusion.ml.models.dinov2 import Dinov2Regressor, default_dinov2_transform
from occlusion.ml.run_config import TrainConfig, load_config


def main():
    p = argparse.ArgumentParser(description="Predict gender on the test set")
    p.add_argument("--run", required=True, help="experiment name under artifacts/models/")
    p.add_argument("--tta", action="store_true", help="average over horizontal flip")
    args = p.parse_args()

    exp_dir = os.path.join(PATHS.models_dir, args.run)
    cfg: TrainConfig = load_config(None, json.load(open(os.path.join(exp_dir, "config.json"))))
    device = cfg.resolve_device()

    model = Dinov2Regressor(
        model_name=cfg.model_type, finetune_mode=cfg.finetune_mode,
        hidden_dim=cfg.hidden_dim, dropout=cfg.head_dropout,
        lora_r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout, lora_targets=cfg.lora_targets).to(device)
    model.load_state_dict(torch.load(os.path.join(exp_dir, "model_best.pth"),
                                     map_location=device))

    catalog = utils.load_catalog_csv(PATHS.test_catalog_path)
    preds = predict_gender(
        model, catalog, device, default_dinov2_transform(cfg.img_size),
        batch_size=max(cfg.batch_size, 128), num_workers=cfg.num_workers,
        tta=args.tta)

    out_path = os.path.join(exp_dir, "gender_pred.csv")
    preds.to_csv(out_path, index=False)
    share_g1 = float((preds["p_gender1"] >= 0.5).mean())
    print(f"wrote {out_path} ({len(preds)} rows) | predicted gender-1 share: {share_g1:.3f}")


if __name__ == "__main__":
    main()
