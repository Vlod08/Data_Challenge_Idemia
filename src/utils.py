import numpy as np
import pandas as pd
from pathlib import Path
from typing import Literal

Mode = Literal["Train", "Test"]

def load_catalog_csv(filename: Path='../data/occlusion_datasets/train.csv', delimiter=','):

    df = pd.read_csv(filename, delimiter=delimiter)

    return df


def print_args(args):
    """Pretty-print all parsed arguments."""
    print("\n\n Parameters of the current run: \n")
    print("-" * 40) 
    for arg, value in vars(args).items():
        print(f"{arg:20}: {value}")
    print("-" * 40)



if __name__ == '__main__':
    print(load_catalog_csv())