import argparse
from sklearn import semi_supervised
from sklearn import metrics
from sklearn import preprocessing
import pandas as pd
import numpy as np
from scipy.io import arff
import os
from os import listdir
from os.path import isfile, join
import random


def run_LabelPropagation(train_X, train_Y, test_X, test_Y, kernel='rbf', gamma=20, n_neighbors=7, max_iter=1000):
    clf = semi_supervised.LabelPropagation(kernel=kernel, gamma=gamma, n_neighbors=n_neighbors, max_iter=max_iter)

    clf.fit(train_X, train_Y)

    accuracy = metrics.accuracy_score(clf.predict(test_X), test_Y)

    return accuracy


def run_LabelSpreading(train_X, train_Y, test_X, test_Y, kernel='rbf', gamma=20, n_neighbors=7, alpha=0.2, max_iter=30):
    clf = semi_supervised.LabelSpreading(kernel=kernel, gamma=gamma, n_neighbors=n_neighbors, alpha=alpha,
                                         max_iter=max_iter)

    clf.fit(train_X, train_Y)

    accuracy = metrics.accuracy_score(clf.predict(test_X), test_Y)

    return accuracy


def run(folder, paramsFolder, output, supervision):
    testFolder = folder.replace("_Train", "_Test")
    files = [f for f in listdir(folder) if isfile(join(folder, f))]
    files = sorted(files)

    spreading_acc = []

    max_values_spr = []
    index_set_spr = []
    mean_value_spr = []
    std_value_spr = []

    propagation_acc = []

    max_values_prop = []
    index_set_prop = []
    mean_value_prop = []
    std_value_prop = []

    datasetNames = []

    for file in files:
        if file.endswith(".arff") and "sup_" not in file:
            spreading_acc.append([])
            propagation_acc.append([])

            train_X, meta_trainX = arff.loadarff(open(join(folder, file), 'rb'))
            train_X = pd.DataFrame(train_X)
            train_Y = train_X['class']
            del train_X['class']

            train_X = np.array(train_X)
            train_Y = np.array(train_Y)

            testFile = file.replace("train_", "test_")
            testFile = testFile.replace("sup_", "")

            if not testFolder.endswith("Test") and not testFolder.endswith("Test/"):
                if testFolder.endswith("/"):
                    testFolder = testFolder[:-4]
                else:
                    testFolder = testFolder[:-3]

            test_X, meta_testX = arff.loadarff(open(join(testFolder, testFile), 'rb'))
            test_X = pd.DataFrame(test_X)
            test_Y = test_X['class']
            del test_X['class']

            test_X = np.array(test_X)
            test_Y = np.array(test_Y)

            alldata = np.append(train_X, test_X, axis=0)
            # scaler = preprocessing.MinMaxScaler()
            # scaler.fit(alldata)
            # alldata = scaler.transform(alldata)

            train_X = alldata[:len(train_X)]
            test_X = alldata[len(train_X):]

            params = open(paramsFolder, 'r')
            params = np.array(params.readlines())

            for paramsSet in range(0, len(params), 9):
                kernel_spreading = getKernel(int(params[paramsSet]))
                gamma_spreading = float(params[paramsSet + 1])
                n_neighbors_spreading = int(params[paramsSet + 2])
                alpha_spreading = float(params[paramsSet + 3])
                max_iter_spreading = int(params[paramsSet + 4])

                rng = np.random.RandomState(random.randint(1, 200000))
                random_unlabeled_points = rng.rand(len(train_Y)) > supervision
                labels_spread = np.copy(train_Y)
                labels_spread[random_unlabeled_points] = str(-1)
                spreading_acc[len(spreading_acc) - 1].append(run_LabelSpreading(train_X, labels_spread, test_X, test_Y,
                                                                                kernel=kernel_spreading,
                                                                                gamma=gamma_spreading,
                                                                                n_neighbors=n_neighbors_spreading,
                                                                                alpha=alpha_spreading,
                                                                                max_iter=max_iter_spreading))

                kernel_propagation = getKernel(int(params[paramsSet + 5]))
                gamma_propagation = float(params[paramsSet + 6])
                n_neighbors_propagation = int(params[paramsSet + 7])
                max_iter_propagation = int(params[paramsSet + 8])

                rng = np.random.RandomState(random.randint(1, 200000))
                random_unlabeled_points = rng.rand(len(train_Y)) > supervision
                labels_prop = np.copy(train_Y)
                labels_prop[random_unlabeled_points] = str(-1)
                propagation_acc[len(propagation_acc) - 1].append(
                    run_LabelPropagation(train_X, labels_prop, test_X, test_Y,
                                         kernel=kernel_propagation,
                                         gamma=gamma_propagation,
                                         n_neighbors=n_neighbors_propagation,
                                         max_iter=max_iter_propagation))

            max_values_spr.append(np.nanmax(spreading_acc[len(spreading_acc) - 1]))
            index_set_spr.append(np.nanargmax(spreading_acc[len(spreading_acc) - 1]))

            mean_value_spr.append(np.nanmean(spreading_acc[len(spreading_acc) - 1]))
            std_value_spr.append(np.nanstd(spreading_acc[len(spreading_acc) - 1], ddof=1))

            max_values_prop.append(np.nanmax(propagation_acc[len(propagation_acc) - 1]))
            index_set_prop.append(np.nanargmax(propagation_acc[len(propagation_acc) - 1]))

            mean_value_prop.append(np.nanmean(propagation_acc[len(propagation_acc) - 1]))
            std_value_prop.append(np.nanstd(propagation_acc[len(propagation_acc) - 1], ddof=1))

            datasetNames.append(file[:-5])
            outputText = "{0}\nPropagation: {1}({2})[{3}]\nSpreading: {4}({5})[{6}]\n\n".format(file,
                                                                                                np.nanmean(propagation_acc[len(propagation_acc) - 1]),
                                                                                                np.nanstd(propagation_acc[len(propagation_acc) - 1], ddof=1),
                                                                                                np.nanargmax(propagation_acc[len(propagation_acc) - 1]),
                                                                                                np.nanmean(spreading_acc[len(spreading_acc) - 1]),
                                                                                                np.nanstd(spreading_acc[len(spreading_acc) - 1], ddof=1),
                                                                                                np.nanargmax(spreading_acc[len(spreading_acc) - 1]))
            print outputText

    writeResults(output, supervision, "label-propagation", propagation_acc, max_values_prop, index_set_prop,
                 mean_value_prop, std_value_prop, datasetNames)
    writeResults(output, supervision, "label-spreading", spreading_acc, max_values_spr, index_set_spr, mean_value_spr,
                 std_value_spr, datasetNames)

def writeResults(outputPath, supervision, method, accs, max_values, index_set, mean_value, std_value,
                 datasetNames):
    if supervision == 1.0:
        outputFile = open(join(outputPath, "{0}-l100.csv".format(method)), 'w+')
    else:
        outputFile = open(
            join(outputPath, "{0}-l{1}.csv".format(method, ('%.2f' % (supervision)).split(".")[1])), 'w+')

    line = "max_value," + ",".join(map(str, max_values)) + "\n"
    line += "index_set," + ",".join(map(str, index_set)) + "\n"
    line += "mean_value," + ",".join(map(str, mean_value)) + "\n"
    line += "std_value," + ",".join(map(str, std_value)) + "\n\n"

    line += "experiment," + ",".join(datasetNames) + "\n"

    for i in range(len(accs[0])):
        line += str(i)
        for j in range(len(datasetNames)):
            line += "," + str(accs[j][i])
        line += "\n"

    outputFile.write(line)

def getKernel(kernel):
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

if not os.path.isdir(output): os.mkdir(output)

run(folder, params, output, supervision)