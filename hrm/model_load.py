import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
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
        self.fc = nn.Linear(7*7*32, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
num_classes=10
# Hyper parameters
num_epochs = 5
num_classes = 10
batch_size = 1
learning_rate = 0.001
model = ConvNet(num_classes).to(device)

pretrained_dict = torch.load('model.ckpt')
model_dict = model.state_dict()

# Modification to the dictionary will go here?

model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)


test_dataset = torchvision.datasets.MNIST(root='../data/',
                                          train=False, 
                                          transform=transforms.ToTensor())

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)


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

