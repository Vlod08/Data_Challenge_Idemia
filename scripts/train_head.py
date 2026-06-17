"""Fast baseline: regression head on frozen DINOv2 embeddings
(pre-extracted by scripts/extract_embeddings.py). Sanity-check pipeline.

  uv run scripts/extract_embeddings.py --model-type dinov2_vits14
  uv run scripts/train_head.py --experiment-name head_v1 \
      --model-type dinov2_vits14
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
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from occlusion import utils
from occlusion.config import PATHS
from occlusion.ml.core import engine
from occlusion.ml.core.losses import build_loss
from occlusion.ml.core.metrics import challenge_score


def build_head(d, hidden=256, dropout=0.2):
    return nn.Sequential(
        nn.LayerNorm(d), nn.Linear(d, hidden), nn.GELU(),
        nn.Dropout(dropout), nn.Linear(hidden, 1))


def parse_args():
    p = argparse.ArgumentParser(description="Train regression head on embeddings")
    p.add_argument("--experiment-name", required=True)
    p.add_argument("--model-type", default="dinov2_vits14")
    p.add_argument("--loss", choices=["balanced", "wmse", "dro"], default="balanced")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    engine.set_seed(args.seed)
    device = engine.get_device()
    utils.print_config(vars(args))

    emb_dir = os.path.join(PATHS.artifacts_dir, "embeddings", args.model_type)
    blob = torch.load(os.path.join(emb_dir, "train.pt"))
    X, y, gen = blob["embeddings"], blob["occ"], blob["gender"]

    df = pd.DataFrame({"g": gen.numpy().astype(int), "y": y.numpy()})
    df["bin"] = pd.qcut(df["y"], q=10, labels=False, duplicates="drop")
    strat = (df["g"].astype(str) + "_" + df["bin"].astype(str)).to_numpy()
    tr, va = train_test_split(np.arange(len(X)), test_size=args.val_split,
                              random_state=args.seed, stratify=strat)

    train_loader = DataLoader(TensorDataset(X[tr], y[tr], gen[tr]),
                              batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X[va], y[va], gen[va]),
                            batch_size=1024, shuffle=False)

    head = build_head(X.shape[1], args.hidden_dim, args.dropout).to(device)
    criterion = build_loss(args.loss)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    exp_dir = os.path.join(PATHS.models_dir, args.experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    best_path = os.path.join(exp_dir, "head_best.pth")
    best, best_m, history = float("inf"), {}, []

    for epoch in range(args.epochs):
        head.train()
        tot, n = 0.0, 0
        for xb, yb, gb in train_loader:
            xb, yb, gb = xb.to(device), yb.to(device), gb.to(device)
            optimizer.zero_grad()
            loss = criterion(torch.sigmoid(head(xb).squeeze(-1)), yb, gb)
            loss.backward()
            optimizer.step()
            tot += loss.item() * xb.size(0)
            n += xb.size(0)

        head.eval()
        preds, ys, gs = [], [], []
        with torch.no_grad():
            for xb, yb, gb in val_loader:
                preds.append(torch.sigmoid(head(xb.to(device)).squeeze(-1)).cpu())
                ys.append(yb)
                gs.append(gb)
        m = challenge_score(torch.cat(preds), torch.cat(ys), torch.cat(gs))
        history.append({"epoch": epoch, "train_loss": tot / n,
                        "val_score": m["score"], "err_g0": m["err_g0"],
                        "err_g1": m["err_g1"]})
        pd.DataFrame(history).to_csv(os.path.join(exp_dir, "history.csv"), index=False)
        print(f"epoch {epoch:03d} | train_loss {tot/n:.5f} | val_score {m['score']:.5f}"
              f" | g0 {m['err_g0']:.5f} | g1 {m['err_g1']:.5f}", flush=True)
        if m["score"] < best:
            best, best_m = m["score"], m
            torch.save(head.state_dict(), best_path)

    hd = pd.DataFrame(history)
    plt.plot(hd["epoch"], hd["train_loss"], label="train loss")
    plt.plot(hd["epoch"], hd["val_score"], label="val score")
    plt.xlabel("epoch")
    plt.legend()
    plt.savefig(os.path.join(exp_dir, "loss_curve.png"))
    plt.close()
    with open(os.path.join(exp_dir, "result.json"), "w") as f:
        json.dump({"best_val_score": best, "err_g0": best_m.get("err_g0"),
                   "err_g1": best_m.get("err_g1")}, f, indent=2)

    head.load_state_dict(torch.load(best_path, map_location=device))
    head.eval()
    tb = torch.load(os.path.join(emb_dir, "test.pt"))
    with torch.no_grad():
        pt = torch.sigmoid(head(tb["embeddings"].to(device)).squeeze(-1)).cpu().numpy()
    pd.DataFrame({"filename": tb["filenames"], "FaceOcclusion": pt, "gender": "x"},
                 columns=["filename", "FaceOcclusion", "gender"]).to_csv(
        os.path.join(exp_dir, "submission.csv"), index=False)

    utils.append_run_index({
        "experiment": args.experiment_name, "model": args.model_type,
        "finetune_mode": "frozen_head", "loss": args.loss, "epochs": args.epochs,
        "best_val_score": round(best, 6),
        "err_g0": round(best_m.get("err_g0", float("nan")), 6),
        "err_g1": round(best_m.get("err_g1", float("nan")), 6),
        "submission": os.path.join(exp_dir, "submission.csv")})
    print(f"best val score: {best:.5f}")


if __name__ == "__main__":
    main()
