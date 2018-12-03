import argparse
import pandas as pd
import numpy as np
from scipy.io import arff
import os
import re
import utils


def create_ordered_arff(arff_file_path, true_file_name, true_file_content, output_path):
    dim = int(re.findall("\d+", true_file_content[0])[0])

    all_indexes = []
    for lines in true_file_content[1:]:
        line = lines.split(" ")[dim:]
        line = map(int, line)

        all_indexes.append(line[1:])

    data, meta = arff.loadarff(open(arff_file_path, 'rb'))
    data = pd.DataFrame(data)

    orderedFile = open(output_path + true_file_name.replace(".true", ".arff"), 'w+')

    labels = data['class'].unique()
    orderedData = pd.DataFrame()
    for label in labels:
        labelData = data[data['class'].isin([label])]
        orderedData = orderedData.append(labelData, ignore_index=True)

    orderedFile.write("@relation {0}\n".format(meta.name))

    for i in xrange(len(meta.names())):
        attr = meta.names()[i]
        if attr != "class":
            orderedFile.write("@attribute {0} {1}\n".format(attr, utils.get_type(meta.types()[i])))
        else:
            orderedFile.write("@attribute {0} {{".format(attr))
            orderedFile.write("{0}".format(",".join(labels)))
            orderedFile.write("}\n")

    orderedFile.write("@data\n")
    data = open(arff_file_path, 'rb+')

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

file_path = args.i
output_path = args.o

file_path = utils.check_directory(file_path)
output_path = utils.check_directory(output_path)

arff_files = []
true_files = []

for file in os.listdir(file_path):
    if file.endswith(".arff"):
        arff_files.append(file)
    elif file.endswith(".true"):
        true_files.append(file)

for true_file_name in true_files:
    arff_file_path = file_path + true_file_name.replace(".true", ".arff")
    if os.path.exists(file_path + true_file_name) and os.path.exists(arff_file_path):
        trueFile = open(file_path + true_file_name, 'rb+')
        trueFileContent = trueFile.readlines()

        create_ordered_arff(arff_file_path=arff_file_path, true_file_name=true_file_name,
                            true_file_content=trueFileContent, output_path=output_path)

        create_ordered_true(trueFileName=true_file_name, trueFileContent=trueFileContent, outputPath=output_path)
