import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets, models
import os
import numpy as np
from config.hyperparameters import Hyperparameters
import argparse
 
np.random.seed(1337) # for reproducibility

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset_name", required=True,help="Dataset Name")
args = vars(ap.parse_args())
dataset_name = args["dataset_name"]
path_to_save = dataset_name

## Import Hyperparameters
param = Hyperparameters(dataset_name=dataset_name)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
								 std=[0.229, 0.224, 0.225])

ds_trans = transforms.Compose([transforms.Scale(299),
							   transforms.CenterCrop(299),
							   transforms.ToTensor(),
							   normalize])

if(dataset_name == "mnist" or dataset_name == "fashion_mnist"):
	ds_trans = transforms.Compose([transforms.Scale(299),
								   transforms.CenterCrop(299),
								   transforms.Grayscale(num_output_channels=3),
								   transforms.ToTensor(),
								   normalize])
	


train_dataset = None
test_dataset = None

if(dataset_name == "mnist"):
	# MNIST Train Dataset
	train_dataset = torchvision.datasets.MNIST(root=param.dataset_path,
												train=True, 
												transform=ds_trans,
												download=True)

	# MNIST Test Dataset
	test_dataset = torchvision.datasets.MNIST(root=param.dataset_path,
												train=False, 
												transform=ds_trans)
elif(dataset_name == "fashion_mnist"):
	# FashionMNIST Train Dataset
	train_dataset = torchvision.datasets.FashionMNIST(root=param.dataset_path,
													train=True, 
													transform=ds_trans,
													download=True)

	# FashionMNIST Test Dataset
	test_dataset = torchvision.datasets.FashionMNIST(root=param.dataset_path,
													train=False, 
													transform=ds_trans)
elif(dataset_name == "svhn"):
	# SVHN Train Dataset
	train_dataset = torchvision.datasets.SVHN(root=param.dataset_path,
											split='train',
											transform=ds_trans,
											download=True)

	# SVHN Test Dataset
	test_dataset = torchvision.datasets.SVHN(root=param.dataset_path,
											split='test',
											transform=ds_trans,
											download=True)

elif(dataset_name == "cifar10"):
	# CIFAR10 Train Dataset
	train_dataset = torchvision.datasets.CIFAR10(root=param.dataset_path,
												train=True, 
												transform=ds_trans,
												download=True)
	
	# CIFAR10 Train Dataset
	test_dataset = torchvision.datasets.CIFAR10(root=param.dataset_path,
												train=False, 
												transform=ds_trans)

elif(dataset_name == "cifar100"):
	# CIFAR100 Train Dataset
	train_dataset = torchvision.datasets.CIFAR100(root=param.dataset_path,
												train=True, 
												transform=ds_trans,
												download=True)
	# CIFAR100 Test Dataset
	test_dataset = torchvision.datasets.CIFAR100(root=param.dataset_path,
												train=False, 
												transform=ds_trans)


### COMMON FOR ALL DATASETS ###
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
									batch_size=param.batch_size, 
									shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
									batch_size=1, 
									shuffle=False)


train_loader_extractor = torch.utils.data.DataLoader(dataset=train_dataset,
									batch_size=1, 
									shuffle=True)

test_loader_extractor = torch.utils.data.DataLoader(dataset=test_dataset,
									batch_size=1, 
									shuffle=False)
################################


# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Import Model
model = None

#resnet18 = models.resnet18(pretrained=True)
#alexnet = models.alexnet(pretrained=True)
#vgg16 = models.vgg16(pretrained=True)
#squeezenet = models.squeezenet1_0(pretrained=True)
#densenet = models.densenet161(pretrained=True)
inception_v3 = True
if(inception_v3 == True):
	model = models.inception_v3(pretrained=True)
	path_to_save = path_to_save + "_imagenet_inception_v3"

model = model.to(device)
model.eval()
train_loader = train_loader_extractor
test_loader  = test_loader_extractor


# Train the model
total_step = len(train_loader)

for param in model.parameters():
	param.requires_grad = False



new_classifier = nn.Sequential(*list(model.fc.children())[:-1])
model.fc = new_classifier


print("Extract Train features")
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

np.savetxt(path_to_save + "_train_features.txt", extract_features,  delimiter=",",fmt='%s')

print("Extract Test features")
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

np.savetxt(path_to_save + "_test_features.txt", extract_features,  delimiter=",",fmt='%s')