"""Rank training runs by a robust metric (plateau mean over the last epochs),
with their key hyperparameters read from config.json. Use after a sweep to
compare above the noise.

  uv run python scripts/summarize.py --prefix sweep_
  uv run python scripts/summarize.py --last 8
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import pandas as pd

from occlusion.config import PATHS

HP_KEYS = ["backbone_lr", "head_lr", "weight_decay", "lora_r", "lora_targets",
           "epochs", "augment", "sampler"]


def _read_hp(run_dir):
    path = os.path.join(run_dir, "config.json")
    if not os.path.exists(path):
        return {}
    c = json.load(open(path))
    out = {k: c.get(k) for k in HP_KEYS}
    if isinstance(out.get("lora_targets"), list):
        out["lora_targets"] = "+".join(out["lora_targets"])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prefix", default="", help="only runs whose name starts with this")
    p.add_argument("--last", type=int, default=5, help="epochs averaged for the plateau")
    args = p.parse_args()

    rows = []
    for hist in glob.glob(str(PATHS.models_dir / "*" / "history.csv")):
        run_dir = os.path.dirname(hist)
        name = os.path.basename(run_dir)
        if not name.startswith(args.prefix):
            continue
        h = pd.read_csv(hist)
        if "val_score" not in h.columns or len(h) == 0:
            continue
        tail = h.tail(args.last)
        best = h.loc[h["val_score"].idxmin()]
        rows.append({
            "run": name, **_read_hp(run_dir),
            "plateau_mean": tail["val_score"].mean(),
            "plateau_std": tail["val_score"].std(),
            "best": best["val_score"], "best_ep": int(best["epoch"]),
        })
    if not rows:
        raise SystemExit(f"no runs found (prefix='{args.prefix}')")

    df = pd.DataFrame(rows).sort_values("plateau_mean").reset_index(drop=True)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print(df.round(6).to_string(index=False))


if __name__ == "__main__":
    main()
