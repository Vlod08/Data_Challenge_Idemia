import pandas as pd
from pathlib import Path
from typing import Literal

from src.config import PATHS

Mode = Literal["Train", "Test"]


def load_catalog_csv(filename: Path = PATHS.train_catalog_path, delimiter: str = ","):
    return pd.read_csv(filename, delimiter=delimiter)


def make_stratify_labels(df: pd.DataFrame, n_bins: int = 10,
                         occ_col: str = "FaceOcclusion", gender_col: str = "gender"):
    occ_bins = pd.qcut(df[occ_col], q=n_bins, labels=False, duplicates="drop")
    return df[gender_col].astype(str) + "_" + occ_bins.astype(str)


def print_args(args):
    print("\n\n Parameters of the current run: \n")
    print("-" * 40)
    for arg, value in vars(args).items():
        print(f"{arg:20}: {value}")
    print("-" * 40)
