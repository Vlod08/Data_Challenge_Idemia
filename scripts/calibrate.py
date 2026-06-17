"""Isotonic calibration of gender-1 occlusion predictions.

The diagnostic shows gender-1 predictions drift below y=x in the high-occlusion
tail (systematic, monotone under-prediction; bias -0.005 -> -0.024 as occ grows).
The metric weight upweights the tail and the level-down floor IS err_g1, so
removing this bias lowers the whole score. We fit a monotone map pred->gt on g1,
weighted by the challenge weight (focuses the fit on the tail).

Fit on a HELD-OUT split (the model's out-of-sample regime). NOT train: there the
in-sample predictions hide the bias (the model partly fit the rare tail), so a
correction fit on train under-corrects on test.

  uv run scripts/calibrate.py \
      --occ-val    artifacts/models/sweep_lora_b_plain_04/occ_val.csv \
      --gender-val artifacts/models/gender_b_lora/gender_val.csv \
      --submission artifacts/models/sweep_lora_b_plain_04/submission.csv \
      --gender-test artifacts/models/gender_b_lora/gender_pred.csv \
      --beta 1.0

Writes occ_val_cal.csv (next to occ-val) and submission_cal.csv (next to the
submission). Then feed BOTH to level_down.py tune to re-tune (t, c) on the
calibrated errors -- err_g1 drops, so the new c* will be smaller.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold

OFFSET = 1.0 / 30.0


def weighted_error(pred, gt, offset=OFFSET):
    w = offset + gt
    return float((w * (pred - gt) ** 2).sum() / max(w.sum(), 1e-12))


def _fit(pred, gt):
    # weight = challenge weight -> the fit concentrates where the metric pays
    return IsotonicRegression(out_of_bounds="clip").fit(
        pred, gt, sample_weight=OFFSET + gt)


def _apply(iso, pred, beta):
    # shrink toward identity: beta=1 full correction, <1 conservative
    return pred + beta * (iso.predict(pred) - pred)


def main():
    p = argparse.ArgumentParser(description="Isotonic g1 tail calibration")
    p.add_argument("--occ-val", required=True, help="filename,gt_occ,gt_gender,pred_occ")
    p.add_argument("--gender-val", required=True, help="filename,...,p_gender1")
    p.add_argument("--submission", required=True, help="test sub: filename,FaceOcclusion")
    p.add_argument("--gender-test", required=True, help="filename,...,p_gender1 (test)")
    p.add_argument("--beta", type=float, default=1.0, help="correction strength in (0,1]")
    p.add_argument("--gender-cut", type=float, default=0.5,
                   help="route the correction to rows with p_gender1 >= cut")
    args = p.parse_args()

    val = pd.read_csv(args.occ_val).merge(
        pd.read_csv(args.gender_val)[["filename", "p_gender1"]], on="filename")
    g1 = val[val.gt_gender == 1.0]
    pred1, gt1 = g1.pred_occ.to_numpy(), g1.gt_occ.to_numpy()

    # honest gain: 5-fold cross-fit on g1 (in-fold scoring would be optimistic)
    oof = pred1.copy()
    for tr, te in KFold(5, shuffle=True, random_state=0).split(pred1):
        oof[te] = _apply(_fit(pred1[tr], gt1[tr]), pred1[te], args.beta)
    e_before = weighted_error(pred1, gt1)
    e_cv = weighted_error(oof, gt1)
    print(f"g1 err  before={e_before:.6f}  cross-fit after={e_cv:.6f}  "
          f"({100 * (e_cv - e_before) / e_before:+.1f}%)  [the honest number]")
    if e_cv >= e_before:
        print("  WARNING: no out-of-sample gain -- calibration overfits, do NOT submit")

    # final map on all g1, applied to rows ROUTED as g1 (predicted gender, as on test)
    iso = _fit(pred1, gt1)
    m = (val.p_gender1 >= args.gender_cut).to_numpy()
    val.loc[m, "pred_occ"] = _apply(iso, val.pred_occ.to_numpy()[m], args.beta)
    out_val = os.path.join(os.path.dirname(args.occ_val), "occ_val_cal.csv")
    val[["filename", "gt_occ", "gt_gender", "pred_occ"]].to_csv(out_val, index=False)
    print(f"wrote {out_val}  ({m.sum()} of {len(val)} rows calibrated)")

    # apply to the test submission, same routing
    sub = pd.read_csv(args.submission).merge(
        pd.read_csv(args.gender_test)[["filename", "p_gender1"]], on="filename", how="left")
    if sub.p_gender1.isna().any():
        raise SystemExit("some submission rows have no gender prediction")
    mt = (sub.p_gender1 >= args.gender_cut).to_numpy()
    occ = sub.FaceOcclusion.to_numpy().copy()
    occ[mt] = np.clip(_apply(iso, occ[mt], args.beta), 0.0, 1.0)
    out_sub = os.path.join(os.path.dirname(args.submission), "submission_cal.csv")
    pd.DataFrame({"filename": sub.filename, "FaceOcclusion": occ,
                  "gender": "x"}).to_csv(out_sub, index=False)
    print(f"wrote {out_sub}  ({mt.sum()} of {len(sub)} test rows calibrated)")
    print("\nnext: level_down.py tune --occ-val occ_val_cal.csv "
          "--submission submission_cal.csv  (re-tune t,c on the calibrated errors)")


if __name__ == "__main__":
    main()
