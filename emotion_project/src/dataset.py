import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class FERDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pixels = np.array(list(map(int, row["pixels"].split())), dtype=np.uint8)
        image = pixels.reshape(48, 48)
        image = Image.fromarray(image)
        label = int(row["emotion"])

        if self.transform:
            image = self.transform(image)

        return image, label