import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from model.mnist_cnn import ConvNet

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = ConvNet(num_classes).to(device)

pretrained_dict = torch.load('model.ckpt')
model_dict = model.state_dict()

# Modification to the dictionary will go here?

model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)


# remove last fully-connected layer
new_classifier = nn.Sequential(*list(model.fc.children())[:-1])
model.fc = new_classifier


# Test the model
#model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
extract_features = []
with torch.no_grad():
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        total = total+1
        print(total)
        #print(outputs.shape)
        for value in outputs:
        	#print(value)
        	#print(value.to('cpu').data.numpy())
        	#print(labels)
        	v = value.to('cpu').data.numpy().tolist()
        	label = labels.to('cpu').data.numpy().tolist()
        	v.append(int(label[0]))
        	extract_features.append(v)
        	
        	#print(extract_features)
        	##labels
        	#input()

import numpy as np
np.savetxt('./file_features.txt', extract_features,  delimiter=",",fmt='%s')

