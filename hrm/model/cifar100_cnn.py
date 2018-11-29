## Model based on Alexnet
'''AlexNet for CIFAR100. FC layers are removed. Paddings are adjusted.
Without BN, the start learning rate should be 0.01
(c) YANG, Wei 
'''

import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from config.hyperparameters import Hyperparameters

class Cifar100ConvNet(nn.Module):
    def __init__(self, num_classes=100):
        super(Cifar100ConvNet, self).__init__()
        # Import Hyperparameters
        param = Hyperparameters()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(256, param.num_classes)


        # MNIST Train Dataset
        self.train_dataset = torchvision.datasets.CIFAR100(root=param.dataset_path,
                                            train=True, 
                                            transform=transforms.ToTensor(),
                                            download=True)

        # MNIST Test Dataset
        self.test_dataset = torchvision.datasets.CIFAR100(root=param.dataset_path,
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
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def getTrainLoader(self):
        return self.train_loader
    
    def getTestLoader(self):
        return self.test_loader


    def getTrainLoaderExtractor(self):
        return self.train_loader_extractor
    
    def getTestLoaderExtractor(self):
        return self.test_loader_extractor