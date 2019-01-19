import argparse
from sklearn import semi_supervised
from sklearn import metrics
import pandas as pd
import numpy as np
from scipy.io import arff
import os
from os import listdir
from os.path import isfile, join
import utils


def run_label_propagation(train_x, train_y, test_x, test_y, kernel='rbf', gamma=20, n_neighbors=7, max_iter=1000):
    clf = semi_supervised.LabelPropagation(kernel=kernel, gamma=gamma, n_neighbors=n_neighbors, max_iter=max_iter)

    clf.fit(train_x, train_y)

    accuracy = metrics.accuracy_score(clf.predict(test_x), test_y)

    return accuracy


def run(folder, params_folder, output, supervision):
    test_folder = folder.replace("_Train", "_Test")

    files = [f for f in listdir(folder) if isfile(join(folder, f)) and f.endswith(".arff") and not f.startswith(".")]

    files = sorted(files)

    propagation_acc = []

    max_values_prop = []
    index_set_prop = []
    mean_value_prop = []
    std_value_prop = []

    dataset_names = []

    for file in files:
        if file.endswith(".arff") and "sup_" not in file:
            propagation_acc.append([])

            train_x, meta_train_x = arff.loadarff(open(join(folder, file), 'rb'))
            train_x = pd.DataFrame(train_x)
            train_y = train_x['class']
            del train_x['class']

            train_x = np.array(train_x)
            train_y = np.array(train_y)

            test_file = file.replace("train_", "test_")
            test_file = test_file.replace("sup_", "")

            if not test_folder.endswith("Test") and not test_folder.endswith("Test/"):
                if test_folder.endswith("/"):
                    test_folder = test_folder[:-4]
                else:
                    test_folder = test_folder[:-3]

            test_x, meta_test_x = arff.loadarff(open(join(test_folder, test_file), 'rb'))
            test_x = pd.DataFrame(test_x)
            test_y = test_x['class']
            del test_x['class']

            test_x = np.array(test_x)
            test_y = np.array(test_y)

            params = open(params_folder, 'r')
            params = np.array(params.readlines())

            for paramsSet in range(0, len(params), 4):
                kernel_propagation = get_kernel(int(params[paramsSet]))
                gamma_propagation = float(params[paramsSet + 1])
                n_neighbors_propagation = int(params[paramsSet + 2])
                max_iter_propagation = int(params[paramsSet + 3])

                unlabeled_points = train_y == '999'
                labels_prop = np.copy(train_y)
                labels_prop[unlabeled_points] = str(-1)
                propagation_acc[len(propagation_acc) - 1].append(
                    run_label_propagation(train_x, labels_prop, test_x, test_y,
                                          kernel=kernel_propagation,
                                          gamma=gamma_propagation,
                                          n_neighbors=n_neighbors_propagation,
                                          max_iter=max_iter_propagation))

            max_values_prop.append(np.max(propagation_acc[len(propagation_acc) - 1]))
            index_set_prop.append(np.argmax(propagation_acc[len(propagation_acc) - 1]))

            mean_value_prop.append(np.mean(propagation_acc[len(propagation_acc) - 1]))
            std_value_prop.append(np.std(propagation_acc[len(propagation_acc) - 1], ddof=1))

            dataset_names.append(file[:-5])

            print "{0}\nPropagation: {1}({2})[{3}]\n\n".format(file,
                                                               np.mean(propagation_acc[len(propagation_acc) - 1]),
                                                               np.std(propagation_acc[len(propagation_acc) - 1],
                                                                      ddof=1),
                                                               np.argmax(propagation_acc[len(propagation_acc) - 1]))

    utils.write_general_results(output, supervision, "label-propagation", propagation_acc, max_values_prop,
                                index_set_prop, mean_value_prop, std_value_prop, dataset_names)


def get_kernel(kernel):
    if kernel == 1:
        return 'rbf'
    else:
        return 'knn'


parser = argparse.ArgumentParser()
parser.add_argument('-i', help='Train Data Directory', required=True)
parser.add_argument('-p', help='Parameters', required=True)
parser.add_argument('-o', help='Output', required=True)
parser.add_argument('-s', help='Percentage of Supervision', required=True, type=float)
args = parser.parse_args()

folder = args.i
params = args.p
output = args.o
supervision = args.s

if not os.path.isdir(output):
    os.mkdir(output)

run(folder, params, output, supervision)
