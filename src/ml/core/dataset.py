from pathlib import Path
from typing import Tuple, Optional, Callable

import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image

from src.config import PATHS
from src.utils import Mode


class Image_Dataset(Dataset):
    def __init__(
            self,
            catalog: pd.DataFrame,
            crops_dir: Path = PATHS.crops_dir,
            img_size: Tuple[int, int] = (224, 224),
            mode: Mode = "Train",
            transform: Optional[Callable] = None
    ):
        super().__init__()
        self.catalog = catalog.reset_index(drop=True)
        self.crops_dir = Path(crops_dir)
        self.img_size = img_size
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return len(self.catalog)

    def _load_image(self, filename: str) -> torch.Tensor:
        img = Image.open(self.crops_dir / filename).convert("RGB")
        if self.transform is not None:
            return self.transform(img)
        img = img.resize(self.img_size)
        return torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

    def __getitem__(self, index):
        row = self.catalog.iloc[index]
        filename = row["filename"]
        image = self._load_image(filename)
        if self.mode == "Test":
            return image, filename
        occ = torch.tensor(float(row["FaceOcclusion"]), dtype=torch.float32)
        gender = torch.tensor(float(row["gender"]), dtype=torch.float32)
        return image, occ, gender


if __name__ == "__main__":
    from src.utils import load_catalog_csv
    cat = load_catalog_csv(PATHS.test_catalog_path)
    ds = Image_Dataset(catalog=cat, mode="Test")
    print(ds[0])
