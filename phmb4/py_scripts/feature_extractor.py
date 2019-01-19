import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets, models
import os
import numpy as np
import argparse


def run(dataset_name, network_name, output_path):
    path_to_save = dataset_name

    ## Import Hyperparameters
    param = Hyperparameters(dataset_name=dataset_name)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    ds_trans = transforms.Compose([transforms.Scale(224), transforms.CenterCrop(224), transforms.ToTensor(), normalize])

    if dataset_name == "mnist" or dataset_name == "fashion_mnist":
        ds_trans = transforms.Compose([transforms.Scale(224), transforms.CenterCrop(224),
                                       transforms.Grayscale(num_output_channels=3), transforms.ToTensor(), normalize])
    train_dataset = None
    test_dataset = None

    if dataset_name == "mnist":
        # MNIST Train Dataset
        train_dataset = torchvision.datasets.MNIST(root=param.dataset_path,
                                                   train=True,
                                                   transform=ds_trans,
                                                   download=True)

        # MNIST Test Dataset
        test_dataset = torchvision.datasets.MNIST(root=param.dataset_path,
                                                  train=False,
                                                  transform=ds_trans)
    elif dataset_name == "fashion_mnist":
        # FashionMNIST Train Dataset
        train_dataset = torchvision.datasets.FashionMNIST(root=param.dataset_path,
                                                          train=True,
                                                          transform=ds_trans,
                                                          download=True)

        # FashionMNIST Test Dataset
        test_dataset = torchvision.datasets.FashionMNIST(root=param.dataset_path,
                                                         train=False,
                                                         transform=ds_trans)
    elif dataset_name == "svhn":
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

    elif dataset_name == "cifar10":
        # CIFAR10 Train Dataset
        train_dataset = torchvision.datasets.CIFAR10(root=param.dataset_path,
                                                     train=True,
                                                     transform=ds_trans,
                                                     download=True)

        # CIFAR10 Train Dataset
        test_dataset = torchvision.datasets.CIFAR10(root=param.dataset_path,
                                                    train=False,
                                                    transform=ds_trans)

    elif dataset_name == "cifar100":
        # CIFAR100 Train Dataset
        train_dataset = torchvision.datasets.CIFAR100(root=param.dataset_path,
                                                      train=True,
                                                      transform=ds_trans,
                                                      download=True)
        # CIFAR100 Test Dataset
        test_dataset = torchvision.datasets.CIFAR100(root=param.dataset_path,
                                                     train=False,
                                                     transform=ds_trans)

    # COMMON FOR ALL DATASETS #
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

    if network_name == "inception-v3":
        model = models.inception_v3(pretrained=True)
    elif network_name == "resnet18":
        model = models.resnet18(pretrained=True)
    elif network_name == "alexnet":
        model = models.alexnet(pretrained=True)
    elif network_name == "vgg16":
        model = models.vgg16(pretrained=True)
    elif network_name == "squeezenet":
        model = models.squeezenet1_0(pretrained=True)
    elif network_name == "densenet161":
        model = models.densenet161(pretrained=True)

    model = model.to(device)
    model.eval()
    train_loader = train_loader_extractor
    test_loader = test_loader_extractor

    # Train the model
    total_step = len(train_loader)

    for param in model.parameters():
        param.requires_grad = False

    new_classifier = nn.Sequential(*list(model.classifier.children())[:-1])
    model.classifier = new_classifier

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


np.random.seed(1337)  # for reproducibility

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", required=True, help="Dataset Name")
parser.add_argument("-n", "--network", required=True, help="Network Name")
parser.add_argument('-o', "--output", required=True, help='Output')
args = parser.parse_args()

dataset_name = args.dataset
network_name = args.network
output_path = args.output

run(dataset_name, network_name, output_path)
