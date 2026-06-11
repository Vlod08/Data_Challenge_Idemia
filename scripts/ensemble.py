"""Ensemble submissions by averaging FaceOcclusion per filename.

  uv run python scripts/ensemble.py --glob 'kfold_lora_b_plain_f*'
  uv run python scripts/ensemble.py --runs lora_b_plain lora_l_plain --method median
"""
from __future__ import annotations

import argparse
import glob
import os

import pandas as pd

from occlusion.config import PATHS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="*", default=[], help="experiment names to ensemble")
    p.add_argument("--glob", help="glob over experiment-dir names, e.g. 'kfold_*_f*'")
    p.add_argument("--method", choices=["mean", "median"], default="mean")
    p.add_argument("--out", default="ensemble", help="output experiment name")
    args = p.parse_args()

    names = list(args.runs)
    if args.glob:
        names += [os.path.basename(d)
                  for d in glob.glob(str(PATHS.models_dir / args.glob))]
    names = sorted(set(names))
    if len(names) < 2:
        raise SystemExit("need >= 2 runs to ensemble")

    cols = []
    for n in names:
        path = PATHS.models_dir / n / "submission.csv"
        if not path.exists():
            raise SystemExit(f"missing submission: {path}")
        cols.append(pd.read_csv(path).set_index("filename")["FaceOcclusion"].rename(n))
        print("  +", path)

    mat = pd.concat(cols, axis=1)
    if mat.isna().any().any():
        raise SystemExit("submissions do not cover the same filenames")
    agg = mat.median(axis=1) if args.method == "median" else mat.mean(axis=1)

    out_dir = PATHS.models_dir / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "submission.csv"
    pd.DataFrame({"filename": agg.index, "FaceOcclusion": agg.values, "gender": "x"}).to_csv(
        out_path, index=False)
    print(f"\n{args.method} of {len(names)} runs -> {out_path} ({len(agg)} rows)")


if __name__ == "__main__":
    main()
