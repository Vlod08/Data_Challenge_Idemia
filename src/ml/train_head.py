from src.ml.core.metrics import challenge_score
from src.ml.core.losses import WeightedMSELoss
from src import config
import matplotlib.pyplot as plt
import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")


PATHS = config.PATHS


def build_head(d, hidden=256, dropout=0.2):
    return nn.Sequential(
        nn.LayerNorm(d),
        nn.Linear(d, hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, 1),
    )


def main():
    args = config.parse_args()
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")

    emb_dir = os.path.join(PATHS.artifacts_dir, "embeddings", args.model_type)
    blob = torch.load(os.path.join(emb_dir, "train.pt"))
    X, y, gen = blob["embeddings"], blob["occ"], blob["gender"]

    df = pd.DataFrame({"g": gen.numpy().astype(int), "y": y.numpy()})
    df["bin"] = pd.qcut(df["y"], q=10, labels=False, duplicates="drop")
    strat = (df["g"].astype(str) + "_" + df["bin"].astype(str)).to_numpy()
    tr, va = train_test_split(np.arange(len(X)), test_size=args.val_split,
                              random_state=args.random_seed, stratify=strat)

    train_loader = DataLoader(TensorDataset(X[tr], y[tr], gen[tr]),
                              batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X[va], y[va], gen[va]),
                            batch_size=512, shuffle=False)

    head = build_head(X.shape[1]).to(device)
    criterion = WeightedMSELoss()
    optimizer = torch.optim.AdamW(head.parameters(), lr=float(args.lr),
                                  weight_decay=float(args.weight_decay))

    exp_dir = os.path.join(args.output_dir, args.experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    best_path = os.path.join(exp_dir, "head_best.pth")
    best, best_m, history = float("inf"), {}, []

    for epoch in range(args.epochs):
        head.train()
        tot, n = 0.0, 0
        for xb, yb, _ in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(torch.sigmoid(head(xb).squeeze(-1)), yb)
            loss.backward()
            optimizer.step()
            tot += loss.item() * xb.size(0)
            n += xb.size(0)

        head.eval()
        preds, ys, gs = [], [], []
        with torch.no_grad():
            for xb, yb, gb in val_loader:
                preds.append(torch.sigmoid(
                    head(xb.to(device)).squeeze(-1)).cpu())
                ys.append(yb)
                gs.append(gb)
        m = challenge_score(torch.cat(preds), torch.cat(ys), torch.cat(gs))

        history.append({"epoch": epoch, "train_wmse": tot / n, "val_score": m["score"],
                        "err_g0": m["err_g0"], "err_g1": m["err_g1"]})
        pd.DataFrame(history).to_csv(os.path.join(
            exp_dir, "history.csv"), index=False)  # écrit à chaque epoch
        print(f"epoch {epoch:03d} | train_wmse {tot/n:.5f} | val_score {m['score']:.5f} "
              f"| g0 {m['err_g0']:.5f} | g1 {m['err_g1']:.5f}", flush=True)
        if m["score"] < best:
            best, best_m = m["score"], m
            torch.save(head.state_dict(), best_path)

    hd = pd.DataFrame(history)
    plt.plot(hd["epoch"], hd["train_wmse"], label="train wmse")
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
        pt = torch.sigmoid(
            head(tb["embeddings"].to(device)).squeeze(-1)).cpu().numpy()
    pd.DataFrame({"filename": tb["filenames"], "FaceOcclusion": pt, "gender": 0.0},
                 columns=["filename", "FaceOcclusion", "gender"]).to_csv(
        os.path.join(exp_dir, "submission.csv"), index=False)
    print(f"best val score: {best:.5f}")


if __name__ == "__main__":
    main()
