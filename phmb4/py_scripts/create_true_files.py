import argparse
import pandas as pd
import numpy as np
from scipy.io import arff
from os import listdir
from os.path import isfile, join

def create_true_file (path, fileName, data, labels):
    trueFile = open(join(path, fileName), 'w+')

    dim = data.shape[1]

    trueFile.write("DIM={0};{1} TRUE\n".format(dim, fileName[:-5].upper()))

    unique_labels = sorted(map(int, np.unique(labels)))
    for label in unique_labels:
        line = " ".join(map(str, [1] * dim))

        newIndexes = np.where(labels == str(label))[0]

        line += " {0} ".format(len(newIndexes))
        line += " ".join(map(str, newIndexes))
        line += "\n"

        trueFile.write(line)

def create_true_files_from_path(folder):
    files = [f for f in listdir(folder) if isfile(join(folder, f))]
    files = sorted(files)

    for file in files:
        if ".arff" in file:
            data, meta = arff.loadarff(open(join(folder, file), 'rb'))
            data = pd.DataFrame(data)
            labels = data.iloc[:,-1]
            del data[data.columns[len(data.columns) - 1]]

            data = np.array(data)
            labels = np.array(labels)

            create_true_file(folder, "{0}.true".format(file[:-5]), data, labels)

parser = argparse.ArgumentParser()
parser.add_argument('-i', help='Input Directory', required=True)
args = parser.parse_args()

folder = args.i

create_true_files_from_path(folder)