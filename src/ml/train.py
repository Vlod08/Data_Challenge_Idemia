from src.ml.models.dinov2 import Dinov2Regressor, default_dinov2_transform
from src.ml.core.metrics import challenge_score
from src.ml.core.losses import WeightedMSELoss
from src.ml.core.dataset import Image_Dataset
from src import utils
from src import config
import matplotlib.pyplot as plt
import os
import json
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use("Agg")


PATHS = config.PATHS


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


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total, n = 0.0, 0
    for images, occ, _ in loader:
        images, occ = images.to(device), occ.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), occ)
        loss.backward()
        optimizer.step()
        total += loss.item() * images.size(0)
        n += images.size(0)
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, targets, genders = [], [], []
    for images, occ, gender in loader:
        preds.append(model(images.to(device)).cpu())
        targets.append(occ)
        genders.append(gender)
    return challenge_score(torch.cat(preds), torch.cat(targets), torch.cat(genders))


@torch.no_grad()
def generate_submission(model, catalog, device, transform, out_path, batch_size, num_workers):
    ds = Image_Dataset(catalog=catalog, mode="Test", transform=transform)
    loader = DataLoader(ds, batch_size=batch_size,
                        shuffle=False, num_workers=num_workers)
    model.eval()
    rows = []
    for images, filenames in loader:
        out = model(images.to(device)).cpu().numpy()
        rows.extend({"filename": fn, "FaceOcclusion": float(p), "gender": 0.0}
                    for fn, p in zip(filenames, out))
    pd.DataFrame(rows, columns=["filename", "FaceOcclusion", "gender"]).to_csv(
        out_path, index=False)
    return out_path


def save_history(history, exp_dir):
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(os.path.join(exp_dir, "history.csv"), index=False)
    plt.figure()
    plt.plot(hist_df["epoch"], hist_df["train_wmse"], label="train wmse")
    plt.plot(hist_df["epoch"], hist_df["val_score"], label="val score")
    plt.xlabel("epoch")
    plt.legend()
    plt.savefig(os.path.join(exp_dir, "loss_curve.png"))
    plt.close()


def main():
    args = config.parse_args()
    set_seed(args.random_seed)
    device = get_device()

    exp_dir = os.path.join(args.output_dir, args.experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    best_path = os.path.join(exp_dir, "model_best.pth")
    sub_path = os.path.join(exp_dir, "submission.csv")
    utils.print_args(args)
    print(f"device: {device}")

    transform = default_dinov2_transform(img_size=args.img_size[0])

    catalog = utils.load_catalog_csv(PATHS.train_catalog_path)
    strat = utils.make_stratify_labels(catalog)
    train_df, val_df = train_test_split(
        catalog, test_size=args.val_split, random_state=args.random_seed, stratify=strat)

    g = torch.Generator()
    g.manual_seed(args.random_seed)
    train_loader = DataLoader(
        Image_Dataset(catalog=train_df, mode="Train", transform=transform),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        worker_init_fn=seed_worker, generator=g, drop_last=True)
    val_loader = DataLoader(
        Image_Dataset(catalog=val_df, mode="Train", transform=transform),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = Dinov2Regressor(model_name=args.model_type,
                            freeze_backbone=True).to(device)
    criterion = WeightedMSELoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(args.lr), weight_decay=float(args.weight_decay))

    if args.dry_run:
        images, occ, gender = next(iter(train_loader))
        print("dry-run ok:", images.shape, model(images.to(device)).shape)
        return

    best = float("inf")
    best_metrics = {}
    history = []
    for epoch in range(args.epochs):
        tr = train_one_epoch(model, train_loader, criterion, optimizer, device)
        m = evaluate(model, val_loader, device)
        print(f"epoch {epoch:03d} | train_wmse {tr:.5f} | val_score {m['score']:.5f} "
              f"| g0 {m['err_g0']:.5f} | g1 {m['err_g1']:.5f}")
        history.append({"epoch": epoch, "train_wmse": tr, "val_score": m["score"],
                        "err_g0": m["err_g0"], "err_g1": m["err_g1"]})
        if m["score"] < best:
            best = m["score"]
            best_metrics = m
            torch.save(model.state_dict(), best_path)
            print(f"  -> new best {best:.5f}")

    save_history(history, exp_dir)
    with open(os.path.join(exp_dir, "result.json"), "w") as f:
        json.dump({"best_val_score": best,
                   "err_g0": best_metrics.get("err_g0"),
                   "err_g1": best_metrics.get("err_g1")}, f, indent=2)

    model.load_state_dict(torch.load(best_path, map_location=device))
    generate_submission(model, utils.load_catalog_csv(PATHS.test_catalog_path),
                        device, transform, sub_path, args.batch_size, args.num_workers)
    print(f"best val score: {best:.5f}\nsubmission: {sub_path}")


if __name__ == "__main__":
    main()
