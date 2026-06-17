"""Train a binary gender classifier (DINOv2 + head) on the face crops.

Reuses the occlusion data/model pipeline; only the loss (BCE) and the selection
metric (accuracy) differ. The trained model is used to identify, at test time,
the better-predicted gender group for post-hoc score optimisation.

  uv run scripts/train_gender.py --config configs/gender_b_lora.yaml
  uv run scripts/train_gender.py --config configs/gender_b_lora.yaml --debug
  uv run scripts/train_gender.py --config configs/gender_b_lora.yaml \
      --override epochs=10 backbone_lr=1e-4
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader

from occlusion import utils
from occlusion.config import PATHS
from occlusion.ml.core import engine
from occlusion.ml.core.dataset import Image_Dataset
from occlusion.ml.core.gender import BinaryGenderLoss, fit_gender
from occlusion.ml.core.sampler import build_weighted_sampler
from occlusion.ml.models.dinov2 import (Dinov2Regressor,
                                        default_dinov2_transform,
                                        train_dinov2_transform)
from occlusion.ml.run_config import (TrainConfig, config_to_dict, load_config,
                                     parse_overrides)


def parse_args():
    p = argparse.ArgumentParser(description="DINOv2 binary gender classifier")
    p.add_argument("--config", help="Path to a YAML experiment config")
    p.add_argument("--override", nargs="*", default=[],
                   help="Inline overrides, e.g. epochs=10 backbone_lr=1e-4")
    p.add_argument("--debug", action="store_true",
                   help="Tiny local run (subsampled data, few epochs)")
    return p.parse_args()


def apply_debug(cfg: TrainConfig) -> TrainConfig:
    cfg.debug = True
    cfg.epochs = 2
    cfg.batch_size = min(cfg.batch_size, 16)
    cfg.num_workers = 0
    cfg.amp = False
    return cfg


def main():
    args = parse_args()
    cfg = load_config(args.config, parse_overrides(args.override))
    if args.debug:
        cfg = apply_debug(cfg)

    engine.set_seed(cfg.seed)
    device = cfg.resolve_device()
    utils.print_config(config_to_dict(cfg))
    print(f"device: {device}")

    exp_dir = os.path.join(PATHS.models_dir, cfg.experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(config_to_dict(cfg), f, indent=2)

    # --- data (stratified by gender x occ-bin keeps the gender balance in val) ---
    catalog = utils.load_catalog_csv(PATHS.train_catalog_path)
    strat = utils.make_stratify_labels(catalog)
    if cfg.n_folds > 0:
        skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
        train_idx, val_idx = list(skf.split(catalog, strat))[cfg.fold]
        train_df, val_df = catalog.iloc[train_idx], catalog.iloc[val_idx]
    else:
        train_df, val_df = train_test_split(
            catalog, test_size=cfg.val_split, random_state=cfg.seed, stratify=strat)
    if cfg.debug:
        train_df, val_df = train_df.iloc[:512], val_df.iloc[:256]

    train_ds = Image_Dataset(catalog=train_df, mode="Train",
                             transform=train_dinov2_transform(cfg.img_size, cfg.augment))
    val_ds = Image_Dataset(catalog=val_df, mode="Train",
                           transform=default_dinov2_transform(cfg.img_size))

    sampler = None
    if cfg.sampler != "none":
        sampler = build_weighted_sampler(
            train_df, mode=cfg.sampler, occ_power=cfg.occ_power, seed=cfg.seed)

    persistent = cfg.num_workers > 0
    g = torch.Generator().manual_seed(cfg.seed)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, sampler=sampler,
        shuffle=(sampler is None), num_workers=cfg.num_workers,
        worker_init_fn=engine.seed_worker, generator=g, drop_last=True,
        pin_memory=(device == "cuda"), persistent_workers=persistent)
    val_loader = DataLoader(
        val_ds, batch_size=max(cfg.batch_size, 128), shuffle=False,
        num_workers=cfg.num_workers, pin_memory=(device == "cuda"),
        persistent_workers=persistent)

    # --- model (same regressor; its sigmoid output is read as P(gender==1)) ---
    model = Dinov2Regressor(
        model_name=cfg.model_type, finetune_mode=cfg.finetune_mode,
        hidden_dim=cfg.hidden_dim, dropout=cfg.head_dropout,
        lora_r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout, lora_targets=cfg.lora_targets).to(device)
    print("params:", model.trainable_parameter_count())

    criterion = BinaryGenderLoss()
    optimizer = engine.build_optimizer(
        model, backbone_lr=cfg.backbone_lr, head_lr=cfg.head_lr,
        weight_decay=cfg.weight_decay)
    total_steps = (len(train_loader) // max(cfg.grad_accum, 1)) * cfg.epochs
    scheduler = engine.build_scheduler(
        optimizer, total_steps=total_steps, warmup_frac=cfg.warmup_frac,
        kind=cfg.scheduler)

    best, best_metrics, best_path = fit_gender(
        model, train_loader, val_loader, criterion, optimizer, device,
        exp_dir=exp_dir, cfg=cfg, scheduler=scheduler)

    print(f"\nbest val acc: {best:.4f} | auc {best_metrics.get('auc'):.4f}"
          f" | prec_g0 {best_metrics.get('prec_g0'):.4f}\nartifacts: {exp_dir}")
    print("next: uv run scripts/predict_gender.py --run "
          f"{cfg.experiment_name}")


if __name__ == "__main__":
    main()
