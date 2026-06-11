from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Mapping

import pandas as pd

from occlusion.config import PATHS

Mode = Literal["Train", "Test"]


def load_catalog_csv(filename: Path = PATHS.train_catalog_path,
                     delimiter: str = ",") -> pd.DataFrame:
    return pd.read_csv(filename, delimiter=delimiter)


def make_stratify_labels(df: pd.DataFrame, n_bins: int = 10,
                         occ_col: str = "FaceOcclusion",
                         gender_col: str = "gender") -> pd.Series:
    """Stratification key: gender x occlusion-quantile bin."""
    occ_bins = pd.qcut(df[occ_col], q=n_bins, labels=False, duplicates="drop")
    return df[gender_col].astype(str) + "_" + occ_bins.astype(str)


def print_config(cfg: Mapping[str, Any]) -> None:
    print("\nRun configuration:")
    print("-" * 40)
    for key, value in cfg.items():
        print(f"{key:18}: {value}")
    print("-" * 40)


def append_run_index(row: Mapping[str, Any], index_path: Path | None = None) -> Path:
    """Append one summary row to the global runs index (multi-run comparison).
    Tolerant to heterogeneous columns across runs."""
    index_path = Path(index_path or (PATHS.models_dir / "runs_index.csv"))
    index_path.parent.mkdir(parents=True, exist_ok=True)
    row = {"timestamp": datetime.now().isoformat(timespec="seconds"), **dict(row)}
    new = pd.DataFrame([row])
    if index_path.exists():
        new = pd.concat([pd.read_csv(index_path), new], ignore_index=True)
    new.to_csv(index_path, index=False)
    return index_path
