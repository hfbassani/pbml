import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from config.hyperparameters import Hyperparameters


# Convolutional neural network (two convolutional layers)
class MnistConvNet(nn.Module):
    def __init__(self):
        super(MnistConvNet, self).__init__()
        # Import Hyperparameters
        param = Hyperparameters(dataset_name='mnist')

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, 32)
        self.fc1 = nn.Linear(32, param.num_classes)
        
        # MNIST Train Dataset
        self.train_dataset = torchvision.datasets.MNIST(root=param.dataset_path,
                                            train=True, 
                                            transform=transforms.ToTensor(),
                                            download=True)

        # MNIST Test Dataset
        self.test_dataset = torchvision.datasets.MNIST(root=param.dataset_path,
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
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out) 
        return out


    def getTrainLoader(self):
        return self.train_loader
    
    def getTestLoader(self):
        return self.test_loader


    def getTrainLoaderExtractor(self):
        return self.train_loader_extractor
    
    def getTestLoaderExtractor(self):
        return self.test_loader_extractor