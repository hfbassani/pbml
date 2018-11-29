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
    model = Cifar10ConvNet()
elif(args["dataset_name"] == 'cifar100'):
    model = Cifar100ConvNet()

model = model.to(device)
train_loader = model.getTrainLoader()
test_loader  = model.getTestLoader()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=param.learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(param.num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, param.num_epochs, i+1, total_step, loss.item()))

# Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
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
    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
if not os.path.exists(param.model_path):
    os.makedirs(param.model_path)

# Save the model checkpoint
if not os.path.exists(param.model_path + '/' + param.dataset_name):
    os.makedirs(param.model_path + '/' + param.dataset_name)
torch.save(model.state_dict(), param.model_path + '/' + param.dataset_name + '/'+ param.dataset_name + '.ckpt')