import torch

import pandas as pd

from scipy.io import arff
from torch.utils.data import Dataset

from torchvision import datasets, transforms

class ArffDataset(Dataset):

    def __init__(self,load_path = "",arff_path=""):
        data = None
        if(load_path == ""):
            data, meta = arff.loadarff(arff_path)
            data = pd.DataFrame(data, dtype=float)
        else: 
            data = pd.read_csv(load_path, sep=",", header=None)
            data = pd.DataFrame(data, dtype=float)

        self.y = data.iloc[:, -1].values

        self.X = data.iloc[:, :-1]
        
        if(load_path != ""):
            self.y = self.y.astype(int)

    def __getitem__(self, index):
        data = torch.tensor(self.X.iloc[index])
        target = torch.tensor(self.y[index], dtype=torch.long)
        return data, target

    def __len__(self):
        return self.X.shape[0]


class CustomDataset(Dataset):

    def __init__(self):
        

        data = datasets.MNIST('../data', train=True, download=True, transform=transforms.Compose([ transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]))

        #data = pd.DataFrame(data, dtype=float)
        #self.y = data.iloc[:, -1].values
        #self.X = data.iloc[:, :-1]

    def __getitem__(self, index):
        #data = torch.tensor(self.X.iloc[index])
        #target = torch.tensor(self.y[index], dtype=torch.long)
        #return data, target
        print("b")
    def __len__(self):
        #return self.X.shape[0]
        print("b")

if __name__ == "__main__":

    print("Hello")

    teste = ArffDataset(load_path = "./file_features.txt")
