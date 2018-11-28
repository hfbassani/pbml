import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from config.hyperparameters import Hyperparameters


class Cifar10ConvNet(nn.Module):
    def __init__(self):
        super(Cifar10ConvNet, self).__init__()
        # Import Hyperparameters
        param = Hyperparameters()
        self.conv1 = nn.Conv2d(3, 20, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(20, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, param.num_classes)

        # MNIST Train Dataset
        self.train_dataset = torchvision.datasets.CIFAR10(root=param.dataset_path,
                                            train=True, 
                                            transform=transforms.ToTensor(),
                                            download=True)

        # MNIST Test Dataset
        self.test_dataset = torchvision.datasets.CIFAR10(root=param.dataset_path,
                                            train=False, 
                                            transform=transforms.ToTensor())

        # MNIST Data Train Loader
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                            batch_size=param.batch_size, 
                                            shuffle=True)

        # MNIST Data Test Loader
        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                            batch_size=1, 
                                            shuffle=False)


        # MNIST Data Train Loader Extractor
        self.train_loader_extractor = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                            batch_size=1, 
                                            shuffle=True)

        # MNIST Data Test Loader Extractor
        self.test_loader_extractor = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                            batch_size=1, 
                                            shuffle=False)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def getTrainLoader(self):
        return self.train_loader
    
    def getTestLoader(self):
        return self.test_loader


    def getTrainLoaderExtractor(self):
        return self.train_loader_extractor
    
    def getTestLoaderExtractor(self):
        return self.test_loader_extractor


    