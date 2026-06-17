"""Per-(gender x occlusion-bin) diagnosis of a trained model on the val split.

  uv run scripts/diagnose.py --experiment-name lora_b_plain

Reconstructs the model from its config.json, rebuilds the exact val split
(same seed / val_split / stratification), and reports where the weighted error
comes from + a calibration curve (mean prediction vs mean ground-truth).
Runs on cuda / mps / cpu — light enough for a laptop.
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from occlusion import utils
from occlusion.config import PATHS
from occlusion.ml.core import engine
from occlusion.ml.core.dataset import Image_Dataset
from occlusion.ml.models.dinov2 import Dinov2Regressor, default_dinov2_transform

OFFSET = 1.0 / 30.0
BINS = [0.0, 0.05, 0.1, 0.2, 0.3, 1.01]  # interpretable occlusion bins (tail-aware)


def load_model(exp_dir, device):
    cfg = json.load(open(os.path.join(exp_dir, "config.json")))
    model = Dinov2Regressor(
        model_name=cfg["model_type"], finetune_mode=cfg["finetune_mode"],
        hidden_dim=cfg["hidden_dim"], dropout=cfg["head_dropout"],
        lora_r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"], lora_targets=cfg["lora_targets"]).to(device)
    model.load_state_dict(torch.load(
        os.path.join(exp_dir, "model_best.pth"), map_location=device))
    model.eval()
    return model, cfg


@torch.no_grad()
def predict_val(model, cfg, device, num_workers):
    catalog = utils.load_catalog_csv(PATHS.train_catalog_path)
    strat = utils.make_stratify_labels(catalog)
    _, val_df = train_test_split(catalog, test_size=cfg["val_split"],
                                 random_state=cfg["seed"], stratify=strat)
    loader = DataLoader(
        Image_Dataset(val_df, mode="Train",
                      transform=default_dinov2_transform(cfg["img_size"])),
        batch_size=256, shuffle=False, num_workers=num_workers)
    preds, occ, gen = [], [], []
    for img, o, g in loader:
        preds.append(model(img.to(device)).float().cpu())
        occ.append(o)
        gen.append(g)
    return (torch.cat(preds).numpy(), torch.cat(occ).numpy(), torch.cat(gen).numpy())


def diagnose(pred, gt, gender):
    df = pd.DataFrame({"pred": pred, "gt": gt, "gender": gender})
    df["w"] = OFFSET + df["gt"]
    df["se"] = df["w"] * (df["pred"] - df["gt"]) ** 2
    df["bin"] = pd.cut(df["gt"], BINS, right=False)

    rows = []
    for g, gdf in df.groupby("gender"):
        wsum_g, err_g = gdf["w"].sum(), gdf["se"].sum() / gdf["w"].sum()
        for b, bdf in gdf.groupby("bin", observed=True):
            rows.append({
                "gender": int(g), "occ_bin": str(b), "n": len(bdf),
                "mean_gt": bdf["gt"].mean(), "mean_pred": bdf["pred"].mean(),
                "bias_pred_minus_gt": bdf["pred"].mean() - bdf["gt"].mean(),
                "bin_werror": bdf["se"].sum() / bdf["w"].sum(),
                "contrib_to_err_g": bdf["se"].sum() / wsum_g})
        rows.append({
            "gender": int(g), "occ_bin": "ALL", "n": len(gdf),
            "mean_gt": gdf["gt"].mean(), "mean_pred": gdf["pred"].mean(),
            "bias_pred_minus_gt": gdf["pred"].mean() - gdf["gt"].mean(),
            "bin_werror": err_g, "contrib_to_err_g": err_g})
    return pd.DataFrame(rows)


def plot_calibration(tab, path):
    plt.figure(figsize=(6, 6))
    sub = tab[tab.occ_bin != "ALL"]
    for g, gdf in sub.groupby("gender"):
        plt.plot(gdf.mean_gt, gdf.mean_pred, "o-", label=f"gender {g}")
    hi = float(sub.mean_gt.max()) * 1.1
    plt.plot([0, hi], [0, hi], "k--", alpha=.5, label="ideal")
    plt.xlabel("mean GT occlusion (per bin)")
    plt.ylabel("mean predicted")
    plt.title("Calibration per gender")
    plt.legend()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment-name", required=True)
    p.add_argument("--models-dir", default=str(PATHS.models_dir))
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = args.device or engine.get_device()
    exp_dir = os.path.join(args.models_dir, args.experiment_name)
    print(f"device: {device} | exp: {exp_dir}")

    model, cfg = load_model(exp_dir, device)
    pred, gt, gender = predict_val(model, cfg, device, args.num_workers)
    tab = diagnose(pred, gt, gender)

    tab.to_csv(os.path.join(exp_dir, "diagnosis.csv"), index=False)
    plot_calibration(tab, os.path.join(exp_dir, "calibration.png"))

    pd.set_option("display.width", 140, "display.max_columns", 20)
    print(tab.round(6).to_string(index=False))
    totals = tab[tab.occ_bin == "ALL"].set_index("gender")["bin_werror"]
    g0, g1 = totals.get(0, float("nan")), totals.get(1, float("nan"))
    score = (g0 + g1) / 2 + abs(g0 - g1)
    print(f"\nerr_g0={g0:.6f}  err_g1={g1:.6f}  score={score:.6f}")
    print(f"saved: {exp_dir}/diagnosis.csv , calibration.png")


if __name__ == "__main__":
    main()
