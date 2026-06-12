"""Binary gender classifier on the face crops, reusing the occlusion pipeline.

The DINOv2 head already outputs sigmoid(.) in [0, 1]; here it is read as
P(gender == 1). Only the loss (BCE) and the selection metric (accuracy) differ
from the occlusion task, so `engine.train_one_epoch` is reused unchanged: the
loss keeps the (pred, occ, gender) signature and simply ignores `occ`.

The classifier identifies, at test time, which gender group each image belongs
to, so the better-predicted group can be targeted for post-hoc score
optimisation (see scripts/level_down.py).
"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from occlusion.ml.core.dataset import Image_Dataset
from occlusion.ml.core.engine import train_one_epoch
from occlusion.ml.run_config import TrainConfig


class BinaryGenderLoss(nn.Module):
    """BCE on P(gender == 1). Manual (not F.binary_cross_entropy) so it is safe
    under autocast. Signature matches the occlusion criterion so the training
    loop is reused as-is. `pos_weight` up-weights the minority class (gender 0)."""

    def __init__(self, pos_weight: float = 1.0):
        super().__init__()
        self.pos_weight = float(pos_weight)

    def forward(self, pred, occ, gender):
        # fp32 BCE: in fp16 (autocast) the clamp is a no-op -- 1-1e-6 rounds to 1.0
        # and 1e-6 underflows, so log(0) -> -inf -> NaN. Upcast makes it safe.
        p = pred.float().clamp(1e-6, 1 - 1e-6)
        gender = gender.float()
        bce = -(gender * p.log() + (1 - gender) * (1 - p).log())
        if self.pos_weight == 1.0:
            return bce.mean()
        w = torch.where(gender == 0.0,
                        torch.as_tensor(self.pos_weight, device=pred.device),
                        torch.as_tensor(1.0, device=pred.device))
        return (w * bce).mean()


@torch.no_grad()
def evaluate_gender(model, loader, device, threshold: float = 0.5):
    """Return (metrics, probs, targets). `prec_g0` is the precision on the
    gender-0 class -- of the images we label g0, the share that truly is g0;
    it guards the worst group when offsetting predicted-g0 images at test time."""
    model.eval()
    probs, targets = [], []
    for images, _occ, gender in loader:
        probs.append(model(images.to(device)).float().cpu())
        targets.append(gender)
    p = torch.cat(probs).numpy()
    y = torch.cat(targets).numpy()
    pred = (p >= threshold).astype(float)
    acc = float((pred == y).mean())
    try:
        auc = float(roc_auc_score(y, p))
    except ValueError:
        auc = float("nan")
    pred_g0 = pred == 0.0
    prec_g0 = float((y[pred_g0] == 0.0).mean()) if pred_g0.any() else float("nan")
    return {"acc": acc, "auc": auc, "prec_g0": prec_g0}, p, y


def _save_gender_history(history, exp_dir):
    hist = pd.DataFrame(history)
    hist.to_csv(os.path.join(exp_dir, "history.csv"), index=False)
    plt.figure()
    plt.plot(hist["epoch"], hist["acc"], label="val acc")
    plt.plot(hist["epoch"], hist["auc"], label="val auc")
    plt.plot(hist["epoch"], hist["prec_g0"], "--", label="prec g0", alpha=.6)
    plt.xlabel("epoch")
    plt.legend()
    plt.savefig(os.path.join(exp_dir, "acc_curve.png"), dpi=120, bbox_inches="tight")
    plt.close()


def fit_gender(model, train_loader, val_loader, criterion, optimizer, device, *,
               exp_dir, cfg: TrainConfig, scheduler=None):
    """Same loop as engine.fit, but selects the best checkpoint on val accuracy
    (higher is better) instead of the challenge score."""
    os.makedirs(exp_dir, exist_ok=True)
    best_path = os.path.join(exp_dir, "model_best.pth")
    scaler = torch.amp.GradScaler("cuda") if (cfg.amp and device == "cuda") else None

    best, best_metrics, history = -1.0, {}, []
    for epoch in range(cfg.epochs):
        tr = train_one_epoch(model, train_loader, criterion, optimizer, device,
                             cfg, scheduler=scheduler, scaler=scaler, epoch=epoch)
        m, _, _ = evaluate_gender(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": tr, **m})
        _save_gender_history(history, exp_dir)
        msg = (f"epoch {epoch:03d} | loss {tr:.5f} | acc {m['acc']:.4f}"
               f" | auc {m['auc']:.4f} | prec_g0 {m['prec_g0']:.4f}")
        if m["acc"] > best:
            best, best_metrics = m["acc"], m
            torch.save(model.state_dict(), best_path)
            msg += "  -> best"
        print(msg, flush=True)

    with open(os.path.join(exp_dir, "result.json"), "w") as f:
        json.dump({"best_acc": best, **best_metrics}, f, indent=2)
    return best, best_metrics, best_path


@torch.no_grad()
def predict_gender(model, catalog, device, transform, *, batch_size, num_workers,
                   tta: bool = False) -> pd.DataFrame:
    """P(gender == 1) for every image. Output columns: filename, p_gender1."""
    loader = DataLoader(Image_Dataset(catalog=catalog, mode="Test", transform=transform),
                        batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model.eval()
    rows = []
    for images, filenames in loader:
        images = images.to(device)
        out = model(images)
        if tta:
            out = 0.5 * (out + model(torch.flip(images, dims=[3])))
        out = out.float().cpu().numpy()
        rows.extend({"filename": fn, "p_gender1": float(p)}
                    for fn, p in zip(filenames, out))
    return pd.DataFrame(rows, columns=["filename", "p_gender1"])
