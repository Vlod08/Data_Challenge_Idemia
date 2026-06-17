"""Test-time augmentation (hflip) -- the only variance reduction available
without retraining. predict_catalog averages the prediction over the image and
its horizontal flip (label-preserving for occlusion %). This first MEASURES
whether TTA lowers err_g1 on the held-out val (the honest test), then writes a
TTA val csv + test submission to re-feed into level_down.py tune.

  uv run scripts/tta.py --occ-run sweep_lora_b_plain_04

If err_g1 drops on val, re-run level_down on occ_val_tta.csv / submission_tta.csv.
"""
from __future__ import annotations

import argparse
import os

from level_down import _device, _load_run, _val_df, score_np


def main():
    p = argparse.ArgumentParser(description="hflip TTA eval + submission")
    p.add_argument("--occ-run", required=True)
    p.add_argument("--no-test", action="store_true", help="only evaluate on val")
    args = p.parse_args()

    from occlusion import utils
    from occlusion.config import PATHS
    from occlusion.ml.core import engine
    from occlusion.ml.models.dinov2 import default_dinov2_transform

    device = _device()
    model, cfg, run_dir = _load_run(args.occ_run, device)
    tf = default_dinov2_transform(cfg.img_size)
    bs, nw = max(cfg.batch_size, 128), cfg.num_workers

    val = _val_df(cfg)
    gt = val[["filename", "FaceOcclusion", "gender"]].rename(
        columns={"FaceOcclusion": "gt_occ", "gender": "gt_gender"})
    preds = {}
    print(f"run: {args.occ_run}  (val {len(val)} rows)")
    for tta in (False, True):
        pr = engine.predict_catalog(model, val, device, tf,
                                    batch_size=bs, num_workers=nw, tta=tta)
        preds[tta] = pr
        m = gt.merge(pr.rename(columns={"FaceOcclusion": "pred"})[["filename", "pred"]],
                     on="filename")
        s, e0, e1 = score_np(m.pred.to_numpy(), m.gt_occ.to_numpy(),
                             m.gt_gender.to_numpy().astype(float))
        print(f"  {'TTA ' if tta else 'base'}: score={s:.6f}  "
              f"err_g0={e0:.6f}  err_g1={e1:.6f}")

    out = gt.merge(preds[True].rename(columns={"FaceOcclusion": "pred_occ"})[
        ["filename", "pred_occ"]], on="filename")
    pv = os.path.join(run_dir, "occ_val_tta.csv")
    out.to_csv(pv, index=False)
    print(f"wrote {pv}")

    if not args.no_test:
        test = utils.load_catalog_csv(PATHS.test_catalog_path)
        sub = engine.predict_catalog(model, test, device, tf,
                                     batch_size=bs, num_workers=nw, tta=True)
        ps = os.path.join(run_dir, "submission_tta.csv")
        sub.to_csv(ps, index=False)
        print(f"wrote {ps} ({len(sub)} rows)")


if __name__ == "__main__":
    main()
