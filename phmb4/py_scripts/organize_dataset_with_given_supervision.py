import argparse
import pandas as pd
import numpy as np
from scipy.io import arff
import os
import re
import sys
from os.path import isfile, join
import random

def check_directory(filePath):
    if os.path.isdir(filePath):
        if not filePath.endswith("/"):
            return filePath + "/"
    else:
        sys.exit("Invalid directory")

def get_type(type):
    if type == "numeric":
        return "real"

def create_ordered_arff(arffFilePath, filePath, outputPath, supervision_r):

    data, meta = arff.loadarff(open(join(filePath, arffFilePath), 'rb'))
    data = pd.DataFrame(data)

    newFile = open(join(outputPath, arffFilePath), 'w+')

    saved_labels = data['class'].unique()
    labels = data['class'].unique()
    orderedData = pd.DataFrame()
    for label in labels:
        labelData = data[data['class'].isin([label])]
        orderedData = orderedData.append(labelData, ignore_index=True)

    labels = data['class']
    while True:
        rng = np.random.RandomState(random.randint(1, 200000))
        random_unlabeled_points = rng.rand(len(labels)) > supervision_r
        curr_labels = np.copy(labels)
        curr_labels[random_unlabeled_points] = str(999)

        if len(labels) - len(curr_labels[random_unlabeled_points]) > len(labels) * (supervision_r):
            break

    data['class'] = curr_labels
    newFile.write("@relation {0}\n".format(meta.name))


    for i in xrange(len(meta.names())):
        attr = meta.names()[i]
        if attr != "class":
            newFile.write("@attribute {0} {1}\n".format(attr, get_type(meta.types()[i])))
        else:
            newFile.write("@attribute {0} {{".format(attr))
            newFile.write("{0}".format(",".join(saved_labels)))
            newFile.write("}\n")

    newFile.write("@data\n")

    for _, row in data.iterrows():
        newFile.write(",".join(map(str, row)) + "\n")

parser = argparse.ArgumentParser()
parser.add_argument('-i', help='Input Directory', required=True)
parser.add_argument('-o', help='Output Directory', required=True)
parser.add_argument('-s', help='Percentage of Supervision', required=True, type=float)
args = parser.parse_args()

filePath = args.i
outputPath = args.o
supervision = args.s

if not os.path.isdir(args.o): os.mkdir(args.o)

filePath = check_directory(filePath)
outputPath = check_directory(outputPath)

for file in os.listdir(filePath):
    if file.endswith(".arff"):
        create_ordered_arff(arffFilePath=file, filePath=filePath, outputPath=outputPath, supervision_r=supervision)
