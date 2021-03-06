import argparse
import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.model_selection import KFold
from os import listdir
from os.path import isfile, join
from sklearn import preprocessing
import os
import utils


def create_arff_file(path, file_name, data, meta):
    arff_file = open(join(path, file_name), 'w+')

    labels = data['class']
    labels = np.array(labels)

    arff_file.write("@relation {0}\n".format(meta.name))

    for i in xrange(len(meta.names())):
        attr = meta.names()[i]
        if attr != "class":
            arff_file.write("@attribute {0} {1}\n".format(attr, utils.get_type(meta.types()[i])))
        else:
            arff_file.write("@attribute {0} {{".format(attr))
            sorted_labels = sorted(map(int, np.unique(labels)))
            arff_file.write("{0}".format(",".join(map(str, sorted_labels))))
            arff_file.write("}\n")

    arff_file.write("@data\n")

    data = np.array(data)
    for row in xrange(len(data)):
        arff_file.write(",".join(map(str, data[row])) + "\n")


def create_true_file(path, file_name, data, labels):
    true_file = open(join(path, file_name), 'w+')

    dim = data.shape[1]

    true_file.write("DIM={0};{1} TRUE\n".format(dim, file_name[:-5].upper()))

    unique_labels = sorted(map(int, np.unique(labels)))
    for label in unique_labels:
        line = " ".join(map(str, [1] * dim))

        newIndexes = np.where(labels == str(label))[0]

        line += " {0} ".format(len(newIndexes))
        line += " ".join(map(str, newIndexes))
        line += "\n"

        true_file.write(line)


def create_xtimes_kfolds(x_times, k_folds, folder, output_folder, norm_type):
    if not os.path.isdir(output_folder + "_Train"):
        os.mkdir(output_folder + "_Train")

    if not os.path.isdir(output_folder + "_Test"):
        os.mkdir(output_folder + "_Test")

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

            for x in xrange(x_times):
                kf = KFold(n_splits=k_folds, shuffle=True)
                fold = 0
                for train, test in kf.split(data):
                    train_data = data[train]
                    test_data = data[test]

                    if norm_type == 'minmax':
                        min_max_scaler = preprocessing.MinMaxScaler().fit(train_data)
                        train_data = min_max_scaler.transform(train_data)
                        test_data = min_max_scaler.transform(test_data)

                    elif norm_type == 'scaler':
                        scaler = preprocessing.StandardScaler().fit(train_data)
                        train_data = scaler.transform(train_data)
                        test_data = scaler.transform(test_data)

                    train_labels = labels[train]
                    test_labels = labels[test]

                    train_file = pd.DataFrame(train_data)
                    train_file['class'] = train_labels
                    create_arff_file(path=output_folder + "_Train", file_name="train_{0}_x{1}_k{2}.arff".format(file[:-5], x + 1, fold + 1),
                                     data=train_file, meta=meta)
                    create_true_file(path=output_folder + "_Train",
                                     file_name="train_{0}_x{1}_k{2}.true".format(file[:-5], x + 1, fold + 1),
                                     data=train_data, labels=train_labels)

                    test_file = pd.DataFrame(test_data)
                    test_file['class'] = test_labels
                    create_arff_file(path=output_folder + "_Test", file_name="test_{0}_x{1}_k{2}.arff".format(file[:-5], x + 1, fold + 1),
                                     data=test_file, meta=meta)
                    create_true_file(path=output_folder + "_Test",
                                     file_name="test_{0}_x{1}_k{2}.true".format(file[:-5], x + 1, fold + 1),
                                     data=test_data, labels=test_labels)

                    fold += 1


def create_true_files_from_path(folder):
    files = [f for f in listdir(folder) if isfile(join(folder, f))]
    files = sorted(files)

    for file in files:
        if ".arff" in file:
            data, meta = arff.loadarff(open(join(folder, file), 'rb'))
            data = pd.DataFrame(data)
            labels = data.iloc[:, -1]
            del data[data.columns[len(data.columns) - 1]]

            data = np.array(data)
            labels = np.array(labels)

            create_true_file(folder, "{0}.true".format(file[:-5]), data, labels)


parser = argparse.ArgumentParser()
parser.add_argument('-i', help='Input Directory', required=True)
parser.add_argument('-o', help='Output Directory', required=True)
parser.add_argument('-x', help='X Times', required=True, type=int)
parser.add_argument('-k', help='K Folds', required=True, type=int)
parser.add_argument('--norm', help='Normalization Type', required=False, default='scaler')
args = parser.parse_args()

folder = args.i
output_folder = args.o
X = args.x
K = args.k
norm_type = args.norm

create_xtimes_kfolds(x_times=X, k_folds=K, folder=folder, output_folder=output_folder, norm_type=norm_type)
