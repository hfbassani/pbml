import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
from config.hyperparameters import Hyperparameters


class FashionConvNet(nn.Module):
    def __init__(self):
        super(FashionConvNet, self).__init__()
        # Import Hyperparameters
        param = Hyperparameters(dataset_name='fashion_mnist')

        # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
     
        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        
        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        
        self.dropout = nn.Dropout(p=0.5)

        # Fully connected 1 (readout)
        self.fc1 = nn.Linear(512, 128)

        self.fc2 = nn.Linear(128, 32)

        self.fc3 = nn.Linear(32, param.num_classes)

        '''
        1. LOADING DATASET
        '''
        normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                             std=[x/255.0 for x in [63.0, 62.1, 66.7]])

        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])

        # FashionMNIST Train Dataset
        self.train_dataset = torchvision.datasets.FashionMNIST(root=param.dataset_path,
                                            train=True, 
                                            transform=transform,
                                            download=True)

        # FashionMNIST Test Dataset
        self.test_dataset = torchvision.datasets.FashionMNIST(root=param.dataset_path,
                                            train=False, 
                                            transform=transform)

        # FashionMNIST Data Train Loader
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                            batch_size=param.batch_size, 
                                            shuffle=True)

        # FashionMNIST Data Test Loader
        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                            batch_size=1, 
                                            shuffle=False)


        # FashionMNIST Data Train Loader Extractor
        self.train_loader_extractor = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                            batch_size=1, 
                                            shuffle=True)

        # FashionMNIST Data Test Loader Extractor
        self.test_loader_extractor = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                            batch_size=1, 
                                            shuffle=False)

    
    def forward(self, x):
        # Convolution 1
        out = self.cnn1(x)
        out = self.relu1(out)
        
        # Max pool 1
        out = self.maxpool1(out)
        
        # Convolution 2 
        out = self.cnn2(out)
        out = self.relu2(out)
        
        # Max pool 2 
        out = self.maxpool2(out)
        
        # Resize
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        
        # Linear function (readout)
        out = self.fc1(out)
        
        # Linear function (readout)
        out = self.fc2(out)
        
        # Linear function (readout)
        out = self.fc3(out)
        
        return out

    def getTrainLoader(self):
        return self.train_loader
    
    def getTestLoader(self):
        return self.test_loader


    def getTrainLoaderExtractor(self):
        return self.train_loader_extractor
    
    def getTestLoaderExtractor(self):
        return self.test_loader_extractor


    