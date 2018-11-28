import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
from config.hyperparameters import Hyperparameters
from model.mnist_cnn import MnistConvNet

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Import Hyperparameters
param = Hyperparameters()

# Import Model
model = MnistConvNet()
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
new_classifier = nn.Sequential(*list(model.fc.children())[:-1])
model.fc = new_classifier

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