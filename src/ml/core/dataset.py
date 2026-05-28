import array

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd

from src.utils import load_catalog_csv, Mode
from src.config import PATHS
from PIL import Image


class Image_Dataset(Dataset):

    def __init__(self, 
                 catalog: pd.DataFrame, 
                 crops_dir: Path = PATHS.crops_dir,
                 img_size : Tuple = (224,224),
                 mode: Mode = 'Train'):
        super().__init__()
        
        self.img_size = img_size
        self.crops_dir = crops_dir
        self.catalog = catalog
        self.mode = mode
    

    def __getitem__(self, index):

        img = Image.open(self.crops_dir / self.catalog.iloc[index]['filename'])
        
        if img.size != self.img_size:
            img = img.resize(self.img_size)         

        img = np.array(img)

        if self.mode == 'Test':
            return img

        elif self.mode == 'Train': 
            Occ = self.catalog.iloc[index]['FaceOcclusion']
            gen = self.catalog.iloc[index]['gender']

            return torch.from_numpy(img), \
                   torch.tensor(Occ),     \
                   gen 
    
    def __len__(self):
        return len(self.catalog)


if __name__ == "__main__":
    
    img_dt = Image_Dataset(catalog_path=PATHS.test_catalog_path, mode='Test')
    print(img_dt.__getitem__(0)) 
