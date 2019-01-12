# Author: Pedro Braga <phmb4@cin.ufpe.br>.

from som import SSSOM

from datasets import ArffDataset
from torch.utils.data import DataLoader

from os.path import join

import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import random
import numpy as np


def read_lines(file_path):
    if os.path.isfile(file_path):
        data = open(file_path, 'r')
        data = np.array(data.read().splitlines())
    else:
        data = []

    return data


parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--batch-size', type=int, default=1, help='input batch size')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--norm-type', help='Normalize', action='store_true', required=False, default=None)
parser.add_argument('--start-idx', help='Start Index', type=int, required=False, default=0)
parser.add_argument('--stop-idx', help='Stop Index', type=int, required=False, default=-1)

parser.add_argument('-i', help='Input Paths', required=True)
parser.add_argument('-t', help='Test Paths', default="")
parser.add_argument('-r', help='Folder to output results', required=True)
parser.add_argument('-p', help='Parameters', required=True)

parser.add_argument('-n', type=int, help='number of nodes', required=False, default=200)
parser.add_argument('-m', help='Map Path')
parser.add_argument('-s', help='Subspace Clustering', action='store_true', required=False)
parser.add_argument('-f', help='Filter Noise', action='store_true', required=False)
parser.add_argument('-e', help='Run Full Evaluation', action='store_true', required=False)
parser.add_argument('-d', help='Display Map', action='store_true', required=False)
parser.add_argument('-k', help='Keep Map', action='store_true', required=False)

parser.add_argument('-S', help='Train Sorted', action='store_true', required=False)

opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if not os.path.exists(os.path.dirname(opt.r)):
    os.makedirs(os.path.dirname(opt.r))

use_cuda = torch.cuda.is_available() and opt.cuda

if use_cuda:
    torch.cuda.init()

ngpu = int(opt.ngpu)
inputPaths = read_lines(opt.i)
testPaths = read_lines(opt.t)
resultsFolder = opt.r
parameters = read_lines(opt.p)

mapPath = opt.m
isSubspace = opt.s
filterNoise = opt.f
numberOfNodes = opt.n
evaluate = opt.e
displayMap = opt.d
keepMap = opt.k

trainSorted = opt.S

parameters_count = 11

start_idx = opt.start_idx
stop_idx = opt.stop_idx

max_idx = int(len(parameters) / parameters_count)

if stop_idx < 0 or stop_idx > max_idx:
    stop_idx = max_idx

if stop_idx <= start_idx:
    start_idx = 0
    stop_idx = max_idx

if len(testPaths) > 0:
    for i, (train, test) in enumerate(zip(inputPaths, testPaths)):

        train_data = ArffDataset(train)
        test_data = ArffDataset(test)
        test_loader = DataLoader(test_data,
                                 num_workers=opt.workers)

        for paramsSet in range(start_idx * parameters_count, stop_idx * parameters_count, parameters_count):
            sssom = SSSOM(use_cuda=use_cuda,
                          ngpu=ngpu,
                          dim=train_data.X.shape[1],
                          max_node_number=numberOfNodes,
                          no_class=999,
                          a_t=float(parameters[paramsSet]),
                          lp=float(parameters[paramsSet + 1]),
                          dsbeta=float(parameters[paramsSet + 2]),
                          age_wins=int(parameters[paramsSet + 3]) * train_data.X.shape[0],
                          e_b=float(parameters[paramsSet + 4]),
                          e_n=float(parameters[paramsSet + 5]),
                          eps_ds=float(parameters[paramsSet + 6]),
                          minwd=float(parameters[paramsSet + 7]),
                          epochs=int(parameters[paramsSet + 8]),
                          e_push=float(parameters[paramsSet + 9]))

            manualSeed = int(parameters[paramsSet + 10])
            random.seed(manualSeed)
            torch.manual_seed(manualSeed)

            train_loader = DataLoader(train_data,
                                      batch_size=opt.batch_size,
                                      shuffle=not trainSorted,
                                      num_workers=opt.workers)

            if use_cuda:
                torch.cuda.manual_seed_all(manualSeed)
                sssom.cuda()
                cudnn.benchmark = True

            sssom.organization(train_loader)

            fileName = test.split("/")[-1].split(".")[0]

            sssom.write_output(join(resultsFolder,
                                    fileName + "_" + str(int(paramsSet / parameters_count)) + ".results"),
                               sssom.cluster_classify(test_loader, opt.s, opt.f))
