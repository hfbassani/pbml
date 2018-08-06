# Author: Pedro Braga <phmb4@cin.ufpe.br>.

import torch
import pandas as pd

from scipy.io import arff
from torch.utils.data import Dataset


class ArffDataset(Dataset):

    def __init__(self, arff_path):
        data, meta = arff.loadarff(arff_path)
        data = pd.DataFrame(data, dtype=float)

        self.y = data.iloc[:, -1].values

        self.X = data.iloc[:, :-1]

    def __getitem__(self, index):
        data = torch.tensor(self.X.iloc[index])
        target = torch.tensor(self.y[index], dtype=torch.long)
        return data, target

    def __len__(self):
        return self.X.shape[0]