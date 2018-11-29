import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
from config.hyperparameters import Hyperparameters
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
train_loader = model.getTrainLoaderExtractor()
test_loader  = model.getTestLoaderExtractor()

# Train the model
total_step = len(train_loader)

# Save the model checkpoint
if not os.path.exists(param.model_path + '/' + param.dataset_name + '/'+ param.dataset_name + '.ckpt'):
    print("Model .ckpt doesnt exist!")
    exit(0)

# Open trained Model
pretrained_dict = torch.load(param.model_path + '/' + param.dataset_name + '/'+ param.dataset_name + '.ckpt')
model_dict = model.state_dict()

# Modification to the dictionary will go here?
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

# remove last fully-connected layer
if(args["dataset_name"] == 'mnist'):
    new_classifier = nn.Sequential(*list(model.fc.children())[:-1])
    model.fc = new_classifier
elif(args["dataset_name"] == 'fashion_mnist'):
    new_classifier = nn.Sequential(*list(model.fc1.children())[:-1])
    model.fc1 = new_classifier
elif(args["dataset_name"] == 'svhn'):
    new_classifier = nn.Sequential(*list(model.fc3.children())[:-1])
    model.fc3 = new_classifier
elif(args["dataset_name"] == 'cifar10'):
    new_classifier = nn.Sequential(*list(model.fc3.children())[:-1])
    model.fc3 = new_classifier
elif(args["dataset_name"] == 'cifar100'):
    new_classifier = nn.Sequential(*list(model.classifier.children())[:-1])
    model.classifier = new_classifier



# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=param.learning_rate)

print("Extract Train features")
# Extract Train features
extract_features = []
with torch.no_grad():
    total = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        total = total+1
        print(total)
        for value in outputs:
        	v = value.to('cpu').data.numpy().tolist()
        	label = labels.to('cpu').data.numpy().tolist()
        	v.append(int(label[0]))
        	extract_features.append(v)

np.savetxt(param.model_path + '/' + param.dataset_name + '/'+ param.dataset_name + '_train_features.txt', extract_features,  delimiter=",",fmt='%s')

print("Extract Test features")
# Extract Test features
extract_features = []
with torch.no_grad():
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        total = total+1
        print(total)
        for value in outputs:
            v = value.to('cpu').data.numpy().tolist()
            label = labels.to('cpu').data.numpy().tolist()
            v.append(int(label[0]))
            extract_features.append(v)

np.savetxt(param.model_path + '/' + param.dataset_name + '/'+ param.dataset_name + '_test_features.txt', extract_features,  delimiter=",",fmt='%s')