import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets, models
import os
import numpy as np
from config.hyperparameters import Hyperparameters
import argparse
 
np.random.seed(0)


## Import Hyperparameters
param = Hyperparameters(dataset_name='cifar10')#args["dataset_name"])


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


ds_trans = transforms.Compose([transforms.Scale(299),
                               transforms.CenterCrop(299),
                               transforms.ToTensor(),
                               normalize])

# MNIST Train Dataset
train_dataset = torchvision.datasets.CIFAR10(root=param.dataset_path,
                                    train=True, 
                                    transform=ds_trans,#transforms.ToTensor(),
                                    download=True)

# MNIST Test Dataset
test_dataset = torchvision.datasets.CIFAR10(root=param.dataset_path,
                                    train=False, 
                                    transform=ds_trans)#transforms.ToTensor())

# MNIST Data Train Loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                    batch_size=param.batch_size, 
                                    shuffle=True)

# MNIST Data Test Loader
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                    batch_size=1, 
                                    shuffle=False)


# MNIST Data Train Loader Extractor
train_loader_extractor = torch.utils.data.DataLoader(dataset=train_dataset,
                                    batch_size=1, 
                                    shuffle=True)

# MNIST Data Test Loader Extractor
test_loader_extractor = torch.utils.data.DataLoader(dataset=test_dataset,
                                    batch_size=1, 
                                    shuffle=False)

'''
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset_name", required=True,help="Dataset Name")
args = vars(ap.parse_args())
'''

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Import Model
model = None

#resnet18 = models.resnet18(pretrained=True)
#alexnet = models.alexnet(pretrained=True)
#vgg16 = models.vgg16(pretrained=True)
#squeezenet = models.squeezenet1_0(pretrained=True)
#densenet = models.densenet161(pretrained=True)
model = models.inception_v3(pretrained=True)


model = model.to(device)
model.eval()
train_loader = train_loader_extractor#model.getTrainLoaderExtractor()
test_loader  = test_loader_extractor#model.getTestLoaderExtractor()

'''
print(model)

for name, child in model.named_children():
    for name2, params in child.named_parameters():
        print(name, name2)

# To view which layers are freeze and which layers are not freezed:
for name, child in model.named_children():
	for name_2, params in child.named_parameters():
		print(name_2, params.requires_grad)

exit(0)
'''

# Train the model
total_step = len(train_loader)

# Save the model checkpoint
#if not os.path.exists(param.model_path + '/' + param.dataset_name + '/'+ param.dataset_name + '.ckpt'):
#    print("Model .ckpt doesnt exist!")
#    exit(0)

# Open trained Model
#pretrained_dict = torch.load(param.model_path + '/' + param.dataset_name + '/'+ param.dataset_name + '.ckpt')
#model_dict = model.state_dict()
#
## Modification to the dictionary will go here?
#model_dict.update(pretrained_dict)
#model.load_state_dict(model_dict)


for param in model.parameters():
    param.requires_grad = False


#if(args["dataset_name"] == 'cifar10'):
#new_classifier = nn.Sequential(*list(model.fc.children())[:-1])
#model.fc = new_classifier

# Since imagenet as 1000 classes , We need to change our last layer according to the number of classes we have,
#num_ftrs = model.fc.in_features
#model.fc = nn.Linear(num_ftrs, 10)
new_classifier = nn.Sequential(*list(model.fc.children())[:-1])
model.fc = new_classifier

# Loss and optimizer
#criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.Adam(model.parameters(), lr=param.learning_rate)

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

#np.savetxt(param.model_path + '/' + param.dataset_name + '/'+ param.dataset_name + '_train_features.txt', extract_features,  delimiter=",",fmt='%s')
np.savetxt('cifar10_imagenet_train_features.txt', extract_features,  delimiter=",",fmt='%s')

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


#np.savetxt(param.model_path + '/' + param.dataset_name + '/'+ param.dataset_name + '_test_features.txt', extract_features,  delimiter=",",fmt='%s')
np.savetxt('cifar10_imagenet_test_features.txt', extract_features,  delimiter=",",fmt='%s')