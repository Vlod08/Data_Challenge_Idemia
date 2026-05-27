import numpy as np
import pandas as pd
from pathlib import Path

def load_catalog_csv(filename: Path='../data/occlusion_datasets/train.csv', delimiter=','):

    df = pd.read_csv(filename, delimiter=delimiter)

    return df




if __name__ == '__main__':
    print(load_catalog_csv())