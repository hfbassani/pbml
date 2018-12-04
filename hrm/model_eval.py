import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from config.hyperparameters import Hyperparameters
import os
from model.mnist_cnn import MnistConvNet
from model.fashion_mnist_cnn import FashionConvNet
from model.svhn_cnn import SvhnConvNet
from model.cifar10_cnn import Cifar10ConvNet
from model.cifar100_cnn import Cifar100ConvNet
from model.WideResNet import WideResNet

import argparse
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset_name", required=True,help="Dataset Name")
args = vars(ap.parse_args())
 
# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

## Import Hyperparameters
param = Hyperparameters(dataset_name=args["dataset_name"])

# Import Model
model = None

if(args["dataset_name"] == 'mnist'):
    model = MnistConvNet()
elif(args["dataset_name"] == 'fashion_mnist'):
    model = FashionConvNet()
elif(args["dataset_name"] == 'svhn'):
    model = SvhnConvNet()
elif(args["dataset_name"] == 'cifar10'):
    #model = Cifar10ConvNet()
    model  = WideResNet(depth=28, num_classes=10)
elif(args["dataset_name"] == 'cifar100'):
    model = Cifar100ConvNet()
model = model.to(device)
train_loader  = model.getTrainLoader()
test_loader  = model.getTestLoader()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=param.learning_rate)

# Open trained Model
pretrained_dict = torch.load(param.model_path + '/' + param.dataset_name + '/'+ param.dataset_name + '.ckpt')
model_dict = model.state_dict()

# Modification to the dictionary will go here?
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

# Test the model with Train Dataset
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)

f = open(param.model_path + '/' + param.dataset_name + '/log_'+ param.dataset_name + ".txt", "w+")
param.printHyper(f)


with torch.no_grad():
    correct = 0
    total = 0
    cont = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Train Accuracy of the model on the ' + str(total) + ' train images: {}%'.format(100 * correct / total))
    f.write('Train Accuracy of the model on the ' + str(total) + ' train images: {}%'.format(100 * correct / total) + '\n')
# Test the model with Test Dataset
with torch.no_grad():
    correct = 0
    total = 0
    cont = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Test Accuracy of the model on the ' + str(total) + ' test images: {}%'.format(100 * correct / total))
    f.write('Test Accuracy of the model on the ' + str(total) + ' test images: {}%'.format(100 * correct / total) + '\n')
