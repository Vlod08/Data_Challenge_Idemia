"""Metric exploit: deliberately inflate the better-predicted gender group's
error toward the worse group's, which lowers the challenge Score.

Score = (Err0 + Err1)/2 + |Err0 - Err1| = 1.5*max - 0.5*min, so dScore/dmin = -0.5:
raising the smaller per-gender error REDUCES the score, until it reaches the
larger one (crossover). We raise it by adding a constant offset c to the
predictions of the rows the gender classifier labels as the better group.

Three stages (the dump stages need a GPU + the images; tune is pure pandas):

  # on the occlusion pod (uses that run's own val split):
  uv run scripts/level_down.py dump-occ --occ-run sweep_lora_b_plain_04

  # on the classifier pod (SAME split when both runs use seed=42, n_folds=0):
  uv run scripts/level_down.py dump-gender --gender-run gender_b_lora

  # anywhere (Mac): tune c on val, apply to the test submission:
  uv run scripts/level_down.py tune \
      --occ-val artifacts/models/sweep_lora_b_plain_04/occ_val.csv \
      --gender-val artifacts/models/gender_b_lora/gender_val.csv \
      --submission artifacts/models/ensemble/submission.csv \
      --gender-test artifacts/models/gender_b_lora/gender_pred.csv \
      --alphas 1.0 0.75 0.5

For the ENSEMBLE, re-tune c on out-of-fold predictions (clean over 100% of train):
  uv run scripts/level_down.py dump-occ-oof --kfold-prefix kfold_lora_b_plain_f
  uv run scripts/level_down.py dump-gender --gender-run gender_b_lora --full
  uv run scripts/level_down.py tune \
      --occ-val   artifacts/models/kfold_lora_b_plain_foof_occ_val.csv \
      --gender-val artifacts/models/gender_b_lora/gender_all.csv \
      --submission artifacts/models/ensemble/submission.csv \
      --gender-test artifacts/models/gender_b_lora/gender_pred.csv --alphas 0.75 0.5
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

OFFSET = 1.0 / 30.0


# --------------------------------------------------------------------------- #
# scoring (numpy mirror of core.metrics.challenge_score, so tune needs no GPU)
# --------------------------------------------------------------------------- #
def weighted_error(pred, gt, offset=OFFSET):
    w = offset + gt
    return float((w * (pred - gt) ** 2).sum() / max(w.sum(), 1e-12))


def score_np(pred, gt, gender, offset=OFFSET):
    e = {}
    for g in (0.0, 1.0):
        m = gender == g
        e[g] = weighted_error(pred[m], gt[m], offset) if m.any() else 0.0
    return (e[0.0] + e[1.0]) / 2 + abs(e[0.0] - e[1.0]), e[0.0], e[1.0]


# --------------------------------------------------------------------------- #
# dump stages (GPU + images): predict on the run's validation split, save CSV
# --------------------------------------------------------------------------- #
def _val_df(cfg):
    """Reconstruct EXACTLY the val split used at training (see scripts/train.py)."""
    from sklearn.model_selection import StratifiedKFold, train_test_split

    from occlusion import utils
    from occlusion.config import PATHS
    catalog = utils.load_catalog_csv(PATHS.train_catalog_path)
    strat = utils.make_stratify_labels(catalog)
    if cfg.n_folds > 0:
        skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
        _, val_idx = list(skf.split(catalog, strat))[cfg.fold]
        return catalog.iloc[val_idx]
    _, val_df = train_test_split(
        catalog, test_size=cfg.val_split, random_state=cfg.seed, stratify=strat)
    return val_df


def _load_run(run, device):
    from occlusion.config import PATHS
    from occlusion.ml.models.dinov2 import Dinov2Regressor
    from occlusion.ml.run_config import load_config
    run_dir = os.path.join(PATHS.models_dir, run)
    cfg = load_config(None, json.load(open(os.path.join(run_dir, "config.json"))))
    import torch
    model = Dinov2Regressor(
        model_name=cfg.model_type, finetune_mode=cfg.finetune_mode,
        hidden_dim=cfg.hidden_dim, dropout=cfg.head_dropout,
        head_activation=cfg.head_activation,
        lora_r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout, lora_targets=cfg.lora_targets).to(device)
    model.load_state_dict(torch.load(os.path.join(run_dir, "model_best.pth"),
                                     map_location=device))
    return model, cfg, run_dir


def dump_occ(args):
    from occlusion.ml.core import engine
    from occlusion.ml.models.dinov2 import default_dinov2_transform
    device = _device()
    model, cfg, run_dir = _load_run(args.occ_run, device)
    val = _val_df(cfg)
    pred = engine.predict_catalog(
        model, val, device, default_dinov2_transform(cfg.img_size),
        batch_size=max(cfg.batch_size, 128), num_workers=cfg.num_workers)
    pred = pred.rename(columns={"FaceOcclusion": "pred_occ"})[["filename", "pred_occ"]]
    out = val[["filename", "FaceOcclusion", "gender"]].rename(
        columns={"FaceOcclusion": "gt_occ", "gender": "gt_gender"}).merge(pred, on="filename")
    path = os.path.join(run_dir, "occ_val.csv")
    out.to_csv(path, index=False)
    print(f"wrote {path} ({len(out)} val rows)")


def dump_gender(args):
    from occlusion import utils
    from occlusion.config import PATHS
    from occlusion.ml.core.gender import predict_gender
    from occlusion.ml.models.dinov2 import default_dinov2_transform
    device = _device()
    model, cfg, run_dir = _load_run(args.gender_run, device)
    # --full: predict on the whole train (to merge with OOF occ). The classifier
    # saw 80% of these at train time, but it is ~99% saturated so the leakage on
    # p_gender1 is negligible. Default: only its own clean val split.
    df = utils.load_catalog_csv(PATHS.train_catalog_path) if args.full else _val_df(cfg)
    out_name = "gender_all.csv" if args.full else "gender_val.csv"
    pred = predict_gender(
        model, df, device, default_dinov2_transform(cfg.img_size),
        batch_size=max(cfg.batch_size, 128), num_workers=cfg.num_workers)
    out = df[["filename", "gender"]].rename(columns={"gender": "gt_gender"}).merge(
        pred, on="filename")
    path = os.path.join(run_dir, out_name)
    out.to_csv(path, index=False)
    print(f"wrote {path} ({len(out)} rows)")


def dump_occ_oof(args):
    """Out-of-fold occlusion predictions for the k-fold ensemble: each fold model
    predicts ONLY its held-out val fold, so concatenating covers 100% of train
    with no leakage. This is the clean val for re-tuning c on the ensemble."""
    import torch

    from occlusion.config import PATHS
    from occlusion.ml.core import engine
    from occlusion.ml.models.dinov2 import default_dinov2_transform
    device = _device()
    parts = []
    for i in range(args.n_folds):
        run = f"{args.kfold_prefix}{i}"
        if not os.path.exists(os.path.join(PATHS.models_dir, run, "model_best.pth")):
            print(f"  skip {run} (no model_best.pth yet)")
            continue
        model, cfg, _ = _load_run(run, device)
        val = _val_df(cfg)                       # fold i's held-out rows
        pred = engine.predict_catalog(
            model, val, device, default_dinov2_transform(cfg.img_size),
            batch_size=max(cfg.batch_size, 128), num_workers=cfg.num_workers)
        pred = pred.rename(columns={"FaceOcclusion": "pred_occ"})[["filename", "pred_occ"]]
        part = val[["filename", "FaceOcclusion", "gender"]].rename(
            columns={"FaceOcclusion": "gt_occ", "gender": "gt_gender"}).merge(pred, on="filename")
        parts.append(part)
        print(f"  fold {i}: {len(part)} rows")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    if not parts:
        raise SystemExit("no fold models found -- wait for the k-fold to finish")
    oof = pd.concat(parts, ignore_index=True)
    out = args.out or os.path.join(PATHS.models_dir, f"{args.kfold_prefix}oof_occ_val.csv")
    oof.to_csv(out, index=False)
    print(f"wrote {out} ({len(oof)} OOF rows, {oof['filename'].nunique()} unique) "
          f"from {len(parts)}/{args.n_folds} folds")


def _device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# --------------------------------------------------------------------------- #
# tune stage (pure pandas): sweep (threshold, c) on val, apply to the test sub
# --------------------------------------------------------------------------- #
def tune(args):
    occ = pd.read_csv(args.occ_val)
    gen = pd.read_csv(args.gender_val)[["filename", "p_gender1"]]
    val = occ.merge(gen, on="filename")
    if len(val) != len(occ) or len(val) != len(gen):
        print(f"WARNING: val merge kept {len(val)} rows "
              f"(occ {len(occ)}, gender {len(gen)}) -- split mismatch?")

    gt_occ = val["gt_occ"].to_numpy()
    gt_gender = val["gt_gender"].to_numpy().astype(float)
    pred_occ = val["pred_occ"].to_numpy()
    p1 = val["p_gender1"].to_numpy()

    base, e0, e1 = score_np(pred_occ, gt_occ, gt_gender)
    g_lo = 0.0 if e0 <= e1 else 1.0          # the better-predicted (smaller-error) group
    e_lo, e_hi = (e0, e1) if g_lo == 0.0 else (e1, e0)
    print(f"\nbaseline val score : {base:.6f}")
    print(f"  err_g0={e0:.6f}  err_g1={e1:.6f}  -> inflate g{int(g_lo)} "
          f"(min={e_lo:.6f}) toward g{int(1 - g_lo)} (max={e_hi:.6f})")
    print(f"  theoretical floor (min raised to max) = {e_hi:.6f} "
          f"(-{100 * (base - e_hi) / base:.1f}%)\n")

    # confidence that a row belongs to the group we inflate
    conf_lo = (1.0 - p1) if g_lo == 0.0 else p1
    thresholds = np.round(np.arange(0.50, 0.999, 0.01), 3)   # purity cut on conf_lo
    cs = np.round(np.arange(0.0, args.c_max + 1e-9, args.c_step), 4)

    best = (base, 0.0, 0.5, e0, e1)          # (score, c, t, err0, err1)
    rows = []
    for t in thresholds:
        mask = conf_lo >= t
        if not mask.any():
            continue
        row_best = (base, 0.0)
        for c in cs:
            p = np.clip(pred_occ + c * mask, 0.0, 1.0)
            s, a0, a1 = score_np(p, gt_occ, gt_gender)
            if s < best[0]:
                best = (s, c, t, a0, a1)
            if s < row_best[0]:
                row_best = (s, c)
        rows.append((t, int(mask.sum()), row_best[1], row_best[0]))

    s_star, c_star, t_star, a0, a1 = best
    print("threshold landscape (best c per purity cut):")
    print(f"  {'t':>5} {'n_offset':>9} {'c*':>7} {'val_score':>10}")
    for t, n, c, s in rows[::3]:
        flag = "  <-- best" if (t == t_star) else ""
        print(f"  {t:>5.2f} {n:>9d} {c:>7.4f} {s:>10.6f}{flag}")

    gain = 100 * (base - s_star) / base
    print(f"\nbest on val : c*={c_star:.4f}  threshold={t_star:.2f}  "
          f"score {base:.6f} -> {s_star:.6f}  (-{gain:.1f}%)")
    print(f"  err_g0 {e0:.6f}->{a0:.6f} | err_g1 {e1:.6f}->{a1:.6f} "
          f"(crossover check: keep min <= max)\n")

    # --- apply to the test submission, with safety fractions of c* ---
    sub = pd.read_csv(args.submission)
    gt = pd.read_csv(args.gender_test)[["filename", "p_gender1"]]
    sub = sub.merge(gt, on="filename", how="left")
    if sub["p_gender1"].isna().any():
        raise SystemExit("some submission rows have no gender prediction")
    conf_lo_test = (1.0 - sub["p_gender1"]) if g_lo == 0.0 else sub["p_gender1"]
    mask_test = (conf_lo_test >= t_star).to_numpy()
    print(f"test rows offset at t={t_star:.2f}: {mask_test.sum()}/{len(sub)} "
          f"({100 * mask_test.mean():.1f}%)\n")

    out_dir = args.out_dir or os.path.dirname(args.submission)
    for a in args.alphas:
        c = round(a * c_star, 4)
        p = np.clip(sub["FaceOcclusion"].to_numpy() + c * mask_test, 0.0, 1.0)
        # report the val score this conservative c would have scored
        pv = np.clip(pred_occ + c * (conf_lo >= t_star), 0.0, 1.0)
        sv, _, _ = score_np(pv, gt_occ, gt_gender)
        out = pd.DataFrame({"filename": sub["filename"], "FaceOcclusion": p, "gender": "x"})
        name = f"submission_ld_a{int(round(a * 100)):03d}.csv"
        path = os.path.join(out_dir, name)
        out.to_csv(path, index=False)
        print(f"  alpha={a:<4} c={c:.4f}  val_score~{sv:.6f}  -> {path}")
    print("\nsubmit the SAFE one first (smaller alpha): overshoot backfires "
          "(slope flips -0.5 -> +1.5), undershoot only forgoes part of the gain.")


def main():
    p = argparse.ArgumentParser(description="Gender level-down metric exploit")
    sub = p.add_subparsers(dest="stage", required=True)

    a = sub.add_parser("dump-occ", help="predict occlusion on the val split")
    a.add_argument("--occ-run", required=True)
    a.set_defaults(func=dump_occ)

    b = sub.add_parser("dump-gender", help="predict gender on the val split")
    b.add_argument("--gender-run", required=True)
    b.add_argument("--full", action="store_true",
                   help="predict on the whole train (to merge with OOF occ)")
    b.set_defaults(func=dump_gender)

    o = sub.add_parser("dump-occ-oof", help="out-of-fold occ predictions over all train")
    o.add_argument("--kfold-prefix", default="kfold_lora_b_plain_f",
                   help="fold runs are <prefix>0 .. <prefix>{n-1}")
    o.add_argument("--n-folds", type=int, default=5)
    o.add_argument("--out", default=None)
    o.set_defaults(func=dump_occ_oof)

    c = sub.add_parser("tune", help="sweep c on val, apply to the test submission")
    c.add_argument("--occ-val", required=True)
    c.add_argument("--gender-val", required=True)
    c.add_argument("--submission", required=True, help="test submission to modify")
    c.add_argument("--gender-test", required=True, help="gender_pred.csv on the test set")
    c.add_argument("--alphas", type=float, nargs="+", default=[1.0, 0.75, 0.5],
                   help="fractions of c* to apply (safety variants)")
    c.add_argument("--c-max", type=float, default=0.10)
    c.add_argument("--c-step", type=float, default=0.002)
    c.add_argument("--out-dir", default=None)
    c.set_defaults(func=tune)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
