import argparse
import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.model_selection import KFold
from os import listdir
from os.path import isfile, join
import os

parser = argparse.ArgumentParser()
parser.add_argument('-i', help='Input Directory', required=True)
parser.add_argument('-o', help='Output Directory', required=True)
parser.add_argument('-x', help='X Times', required=True, type=int)
parser.add_argument('-k', help='K Folds', required=True, type=int)
args = parser.parse_args()

folder = args.i
outputFolder = args.o
X = args.x
K = args.k

if not os.path.isdir(outputFolder): os.mkdir(outputFolder)
files = [f for f in listdir(folder) if isfile(join(folder, f))]
files = sorted(files)

for file in files:
    if ".arff" in file:
        data, meta = arff.loadarff(open(join(folder, file), 'rb'))
        data = pd.DataFrame(data)
        labels = data['class']
        del data['class']

        data = np.array(data)
        labels = np.array(labels)

        for x in xrange(X):
            kf = KFold(n_splits=K)
            fold = 0
            for train, test in kf.split(data):
                train_data = data[train]
                test_data = data[test]

                train_labels = labels[train]
                test_labels = labels[test]

                trainFile = pd.DataFrame(train_data)
                trainFile['class'] = train_labels
                trainFile.to_csv(join(outputFolder, "tain_{0}_x{1}_k{2}.arff".format(file[:-5], x + 1,fold + 1)), sep=',', encoding='utf-8', index=False, header=None)

                testFile = pd.DataFrame(test_data)
                testFile['class'] = test_labels
                testFile.to_csv(join(outputFolder, "test_{0}_x{1}_k{2}.arff".format(file[:-5], x + 1,fold + 1)), sep=',', encoding='utf-8', index=False, header=None)

                trueTestfile = open(join(outputFolder, "test_{0}_x{1}_k{2}.true".format(file[:-5], x + 1, fold + 1)), 'w+')

                dim = test_data.shape[1]

                trueTestfile.write("DIM={0};{1} TRUE\n".format(dim, file[:-5].upper()))

                unique_labels = sorted(map(int, list(set(test_labels))))
                for label in unique_labels:
                    line = " ".join(map(str, [1] * dim))

                    newIndexes = np.where(test_labels == str(label))[0]

                    line += " "
                    line += " ".join(map(str,newIndexes))
                    line += "\n"

                    trueTestfile.write(line)

                fold += 1