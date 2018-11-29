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
        param = Hyperparameters(dataset_name='cifar10')
        self.conv1 = nn.Conv2d(3, 20, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(20, 16, 5)
        self.conv3 = nn.Conv2d(16, 8, 5,padding=2)
        self.fc1 = nn.Linear(8 * 5 * 5, 100)
        self.fc2 = nn.Linear(100, 64)
        self.fc3 = nn.Linear(64, param.num_classes)

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
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
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


    