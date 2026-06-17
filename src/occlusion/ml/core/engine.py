"""End-to-end training engine (frozen / lora / full). Shared loop so several
runs can reuse it. The best checkpoint is selected on the challenge Score
computed on the validation set (not on the loss)."""
from __future__ import annotations

import json
import math
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from occlusion.ml.core.dataset import Image_Dataset
from occlusion.ml.core.metrics import challenge_score
from occlusion.ml.run_config import TrainConfig

LOG_EVERY = 50


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(_):
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)


def build_optimizer(model, *, backbone_lr, head_lr, weight_decay):
    return torch.optim.AdamW(model.param_groups(
        backbone_lr=backbone_lr, head_lr=head_lr, weight_decay=weight_decay))


def build_scheduler(optimizer, *, total_steps, warmup_frac=0.05, kind="cosine"):
    if kind == "none":
        return None
    warmup_steps = max(1, int(total_steps * warmup_frac))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        if kind == "constant":
            return 1.0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, loader, criterion, optimizer, device, cfg: TrainConfig,
                    *, scheduler=None, scaler=None, epoch: int = 0):
    model.train()
    total, n = 0.0, 0
    optimizer.zero_grad(set_to_none=True)
    nb = len(loader)
    for i, (images, occ, gender) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        occ = occ.to(device, non_blocking=True)
        gender = gender.to(device, non_blocking=True)

        with torch.autocast(device_type=device, enabled=cfg.amp and device == "cuda"):
            pred = model(images)
            loss = criterion(pred, occ, gender) / cfg.grad_accum

        (scaler.scale(loss) if scaler else loss).backward()

        if (i + 1) % cfg.grad_accum == 0 or (i + 1) == nb:
            if cfg.clip_grad > 0:
                if scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        bs = images.size(0)
        total += loss.item() * cfg.grad_accum * bs
        n += bs
        if LOG_EVERY and (i + 1) % LOG_EVERY == 0:
            print(f"  epoch {epoch:03d} | {i+1}/{nb} | loss {total/n:.5f}", flush=True)
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    preds, targets, genders = [], [], []
    for images, occ, gender in loader:
        preds.append(model(images.to(device)).float().cpu())
        targets.append(occ)
        genders.append(gender)
    # clamp to the valid occlusion range (no-op for the sigmoid head, needed for linear)
    return challenge_score(torch.cat(preds).clamp(0, 1), torch.cat(targets), torch.cat(genders))


@torch.no_grad()
def predict_catalog(model, catalog, device, transform, *, batch_size, num_workers,
                    tta: bool = False) -> pd.DataFrame:
    loader = DataLoader(Image_Dataset(catalog=catalog, mode="Test", transform=transform),
                        batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model.eval()
    rows = []
    for images, filenames in loader:
        images = images.to(device)
        out = model(images)
        if tta:  # average prediction over image + horizontal flip
            out = 0.5 * (out + model(torch.flip(images, dims=[3])))
        out = out.clamp(0, 1).float().cpu().numpy()  # no-op for sigmoid, bounds the linear head
        rows.extend({"filename": fn, "FaceOcclusion": float(p), "gender": "x"}
                    for fn, p in zip(filenames, out))
    return pd.DataFrame(rows, columns=["filename", "FaceOcclusion", "gender"])


def save_history(history, exp_dir):
    hist = pd.DataFrame(history)
    hist.to_csv(os.path.join(exp_dir, "history.csv"), index=False)
    plt.figure()
    plt.plot(hist["epoch"], hist["train_loss"], label="train loss")
    plt.plot(hist["epoch"], hist["val_score"], label="val score")
    plt.plot(hist["epoch"], hist["err_g0"], "--", label="err g0", alpha=.6)
    plt.plot(hist["epoch"], hist["err_g1"], "--", label="err g1", alpha=.6)
    plt.xlabel("epoch")
    plt.legend()
    plt.savefig(os.path.join(exp_dir, "loss_curve.png"), dpi=120, bbox_inches="tight")
    plt.close()


def fit(model, train_loader, val_loader, criterion, optimizer, device, *,
        exp_dir, cfg: TrainConfig, scheduler=None):
    os.makedirs(exp_dir, exist_ok=True)
    best_path = os.path.join(exp_dir, "model_best.pth")
    scaler = torch.amp.GradScaler("cuda") if (cfg.amp and device == "cuda") else None

    best, best_metrics, history = float("inf"), {}, []
    for epoch in range(cfg.epochs):
        tr = train_one_epoch(model, train_loader, criterion, optimizer, device,
                             cfg, scheduler=scheduler, scaler=scaler, epoch=epoch)
        m = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": tr, "val_score": m["score"],
                        "err_g0": m["err_g0"], "err_g1": m["err_g1"]})
        save_history(history, exp_dir)
        msg = (f"epoch {epoch:03d} | train_loss {tr:.5f} | val_score {m['score']:.5f}"
               f" | g0 {m['err_g0']:.5f} | g1 {m['err_g1']:.5f}")
        if m["score"] < best:
            best, best_metrics = m["score"], m
            torch.save(model.state_dict(), best_path)
            msg += "  -> best"
        print(msg, flush=True)

    with open(os.path.join(exp_dir, "result.json"), "w") as f:
        json.dump({"best_val_score": best,
                   "err_g0": best_metrics.get("err_g0"),
                   "err_g1": best_metrics.get("err_g1")}, f, indent=2)
    return best, best_metrics, best_path
