"""End-to-end DINOv2 occlusion training (frozen / lora / full).

  uv run python scripts/train.py --config configs/lora_vitb.yaml
  uv run python scripts/train.py --config configs/lora_vitb.yaml --debug
  uv run python scripts/train.py --config configs/lora_vitb.yaml \
      --override epochs=10 backbone_lr=5e-5
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from occlusion import utils
from occlusion.config import PATHS
from occlusion.ml.core import engine
from occlusion.ml.core.dataset import Image_Dataset
from occlusion.ml.core.losses import build_loss
from occlusion.ml.core.sampler import build_weighted_sampler
from occlusion.ml.models.dinov2 import (Dinov2Regressor,
                                        default_dinov2_transform,
                                        train_dinov2_transform)
from occlusion.ml.run_config import (TrainConfig, config_to_dict, load_config,
                                     parse_overrides)


def parse_args():
    p = argparse.ArgumentParser(description="DINOv2 occlusion training")
    p.add_argument("--config", help="Path to a YAML experiment config")
    p.add_argument("--override", nargs="*", default=[],
                   help="Inline overrides, e.g. epochs=10 backbone_lr=5e-5")
    p.add_argument("--debug", action="store_true",
                   help="Tiny local run (subsampled data, few epochs)")
    return p.parse_args()


def apply_debug(cfg: TrainConfig) -> TrainConfig:
    """Shrink a run so it executes in minutes on a laptop GPU (mps/cuda)."""
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

    # --- data ---
    catalog = utils.load_catalog_csv(PATHS.train_catalog_path)
    strat = utils.make_stratify_labels(catalog)
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

    g = torch.Generator().manual_seed(cfg.seed)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, sampler=sampler,
        shuffle=(sampler is None), num_workers=cfg.num_workers,
        worker_init_fn=engine.seed_worker, generator=g, drop_last=True,
        pin_memory=(device == "cuda"))
    val_loader = DataLoader(
        val_ds, batch_size=max(cfg.batch_size, 128), shuffle=False,
        num_workers=cfg.num_workers, pin_memory=(device == "cuda"))

    # --- model ---
    model = Dinov2Regressor(
        model_name=cfg.model_type, finetune_mode=cfg.finetune_mode,
        hidden_dim=cfg.hidden_dim, dropout=cfg.head_dropout,
        lora_r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout, lora_targets=cfg.lora_targets).to(device)
    print("params:", model.trainable_parameter_count())

    criterion = build_loss(cfg.loss)
    optimizer = engine.build_optimizer(
        model, backbone_lr=cfg.backbone_lr, head_lr=cfg.head_lr,
        weight_decay=cfg.weight_decay)
    total_steps = (len(train_loader) // max(cfg.grad_accum, 1)) * cfg.epochs
    scheduler = engine.build_scheduler(
        optimizer, total_steps=total_steps, warmup_frac=cfg.warmup_frac,
        kind=cfg.scheduler)

    best, best_metrics, best_path = engine.fit(
        model, train_loader, val_loader, criterion, optimizer, device,
        exp_dir=exp_dir, cfg=cfg, scheduler=scheduler)

    # --- submission on the best checkpoint (skipped in debug) ---
    sub_path = None
    if not cfg.debug:
        model.load_state_dict(torch.load(best_path, map_location=device))
        sub = engine.predict_catalog(
            model, utils.load_catalog_csv(PATHS.test_catalog_path), device,
            default_dinov2_transform(cfg.img_size),
            batch_size=max(cfg.batch_size, 128), num_workers=cfg.num_workers,
            tta=cfg.tta)
        sub_path = os.path.join(exp_dir, "submission.csv")
        sub.to_csv(sub_path, index=False)

    utils.append_run_index({
        "experiment": cfg.experiment_name, "model": cfg.model_type,
        "finetune_mode": cfg.finetune_mode, "loss": cfg.loss,
        "sampler": cfg.sampler, "augment": cfg.augment, "epochs": cfg.epochs,
        "batch_size": cfg.batch_size, "head_lr": cfg.head_lr,
        "backbone_lr": cfg.backbone_lr, "best_val_score": round(best, 6),
        "err_g0": round(best_metrics.get("err_g0", float("nan")), 6),
        "err_g1": round(best_metrics.get("err_g1", float("nan")), 6),
        "submission": sub_path, "debug": cfg.debug,
    })
    print(f"\nbest val score: {best:.5f}\nartifacts: {exp_dir}")


if __name__ == "__main__":
    main()
