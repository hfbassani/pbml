# Author: Pedro Braga <phmb4@cin.ufpe.br>.

import torch
import pandas as pd

from scipy.io import arff
from torch.utils.data import Dataset


class ArffDataset(Dataset):

    def __init__(self, load_path):

        if load_path.endswith(".arff"):
            data, meta = arff.loadarff(load_path)
            data = pd.DataFrame(data, dtype=float)
        else:
            data = pd.read_csv(load_path, sep=",", header=None)
            data = pd.DataFrame(data, dtype=float)

        self.y = data.iloc[:, -1].values

        self.X = data.iloc[:, :-1]
        
        if load_path != "":
            self.y = self.y.astype(int)

    def __getitem__(self, index):
        data = torch.tensor(self.X.iloc[index])
        target = torch.tensor(self.y[index], dtype=torch.long)
        return data, target

    def __len__(self):
        return self.X.shape[0]
