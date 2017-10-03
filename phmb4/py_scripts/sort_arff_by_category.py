import argparse
import pandas as pd
import numpy as np
from scipy.io import arff
import os
import re
import sys

def check_directory(filePath):
    if os.path.isdir(filePath):
        if not filePath.endswith("/"):
            return filePath + "/"
    else:
        sys.exit("Invalid directory")

def get_type(type):
    if type == "numeric":
        return "real"

def create_ordered_arff(arffFilePath, trueFileName, trueFileContent, outputPath):
    dim = int(re.findall("\d+", trueFileContent[0])[0])

    all_indexes = []
    for lines in trueFileContent[1:]:
        line = lines.split(" ")[dim:]
        line = map(int, line)

        all_indexes.append(line[1:])

    data, meta = arff.loadarff(open(arffFilePath, 'rb'))
    data = pd.DataFrame(data)

    orderedFile = open(outputPath + trueFileName.replace(".true", ".arff"), 'w+')

    labels = data['class'].unique()
    orderedData = pd.DataFrame()
    for label in labels:
        labelData = data[data['class'].isin([label])]
        orderedData = orderedData.append(labelData, ignore_index=True)

    orderedFile.write("@relation {0}\n".format(meta.name))

    for i in xrange(len(meta.names())):
        attr = meta.names()[i]
        if attr != "class":
            orderedFile.write("@attribute {0} {1}\n".format(attr, get_type(meta.types()[i])))
        else:
            orderedFile.write("@attribute {0} {{".format(attr))
            orderedFile.write("{0}".format(",".join(labels)))
            orderedFile.write("}\n")

    orderedFile.write("@data\n")
    data = open(arffFilePath, 'rb+')

    skipLines = 1 + dim + 1 + 1
    arffData = data.readlines()[skipLines:]

    for class_indexes in all_indexes:
        for index in class_indexes:
            orderedFile.write(arffData[index])

def create_ordered_true(trueFileName, trueFileContent, outputPath):
    dim = int(re.findall("\d+", trueFileContent[0])[0])

    orderedFile = open(outputPath + trueFileName, 'w+')

    orderedFile.write(trueFileContent[0])

    count = 0
    for lines in trueFileContent[1:]:
        line = lines.split(" ")

        ammount = int(line[dim:][0])

        newIndexes = np.linspace(start=count, stop=count + ammount - 1, num=ammount, dtype=int)

        newLine = " ".join(line[:dim + 1]) + " " + " ".join(map(str, newIndexes)) + "\n"

        orderedFile.write(newLine)

        count += ammount

parser = argparse.ArgumentParser()
parser.add_argument('-i', help='Input Directory', required=True)
parser.add_argument('-o', help='Output Directory', required=True)
args = parser.parse_args()

filePath = args.i
outputPath = args.o

filePath = check_directory(filePath)
outputPath = check_directory(outputPath)

arffFiles = []
trueFiles = []
for file in os.listdir(filePath):
    if file.endswith(".arff"):
        arffFiles.append(file)
    elif file.endswith(".true"):
        trueFiles.append(file)

for trueFileName in trueFiles:
    arffFilePath = filePath + trueFileName.replace(".true", ".arff")
    if os.path.exists(filePath + trueFileName) and os.path.exists(arffFilePath):
        trueFile = open(filePath + trueFileName, 'rb+')
        trueFileContent = trueFile.readlines()

        create_ordered_arff(arffFilePath=arffFilePath, trueFileName=trueFileName,
                            trueFileContent=trueFileContent, outputPath=outputPath)

        create_ordered_true(trueFileName=trueFileName, trueFileContent=trueFileContent, outputPath=outputPath)
