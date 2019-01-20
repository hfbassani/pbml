import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets, models
import os
import numpy as np
from config.hyperparameters import Hyperparameters
import argparse
import cv2
import numpy as np
from skimage.io import imsave

np.random.seed(1337) # for reproducibility

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset_name", required=True,help="Dataset Name")
ap.add_argument("-o", "--output_path", required=True,help="Output Path")
ap.add_argument("-n", "--n_images", required=True,help="Number of Images", type=int)
args = vars(ap.parse_args())
dataset_name = args["dataset_name"]
output_path = args["output_path"]
n_images = args["n_images"]
path_to_save = dataset_name

## Import Hyperparameters
param = Hyperparameters(dataset_name=dataset_name)

# Save the model checkpoint
if not os.path.exists(path_to_save):
    os.makedirs(path_to_save)


train_dataset = None
h,w,d = 0,0,0
if(dataset_name == "mnist"):
	# MNIST Train Dataset
	train_dataset = torchvision.datasets.MNIST(root=param.dataset_path,
												train=True, 
												transform=transforms.ToTensor(),
												download=True)
	h,w,d = 28,28,1

elif(dataset_name == "fashion_mnist"):
	# FashionMNIST Train Dataset
	train_dataset = torchvision.datasets.FashionMNIST(root=param.dataset_path,
													train=True, 
													transform=transforms.ToTensor(),
													download=True)
	h,w,d = 28,28,1
elif(dataset_name == "svhn"):
	# SVHN Train Dataset
	train_dataset = torchvision.datasets.SVHN(root=param.dataset_path,
											split='train',
											transform=transforms.ToTensor(),
											download=True)
	h,w,d = 32,32,3
elif(dataset_name == "cifar10"):
	# CIFAR10 Train Dataset
	train_dataset = torchvision.datasets.CIFAR10(root=param.dataset_path,
												train=True, 
												transform=transforms.ToTensor(),
												download=True)
	h,w,d = 32,32,3
elif(dataset_name == "cifar100"):
	# CIFAR100 Train Dataset
	train_dataset = torchvision.datasets.CIFAR100(root=param.dataset_path,
												train=True, 
												transform=transforms.ToTensor(),
												download=True)
	# CIFAR100 Test Dataset
	test_dataset = torchvision.datasets.CIFAR100(root=param.dataset_path,
												train=False, 
												transform=transforms.ToTensor())

	h,w,d = 32,32,3
### COMMON FOR ALL DATASETS ###
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
									batch_size=1, 
									shuffle=True)


with torch.no_grad():
	total = 0
	for images, labels in train_loader:
		total = total+1
		
		if(dataset_name == "mnist" or dataset_name == "fashion_mnist"):
			images = images.reshape(h,w).numpy()
		else:
			images = images.reshape(d,h,w).numpy()
			images = np.moveaxis(images,0,-1)
			images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
		
		cv2.imwrite(output_path + "/{}.png".format(total), images*255)

		if(total >= n_images):
			break
		