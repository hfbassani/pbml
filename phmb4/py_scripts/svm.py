import argparse
from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing
import pandas as pd
import numpy as np
from scipy.io import arff
import os
from os import listdir
from os.path import isfile, join

def run_svm(train_X, train_Y, test_X, test_Y, c=1.0, kernel='rbf', degree=3, gamma=0.1):
    clf = svm.SVC(C=c, kernel=kernel, degree=degree, gamma=gamma,
                  coef0=0.0, shrinking=True, probability=False,
                  tol=1e-3, cache_size=200, class_weight=None,
                  verbose=False, max_iter=-1, decision_function_shape=None,
                  random_state=None)

    clf.fit(train_X, train_Y)

    accuracy = metrics.accuracy_score(clf.predict(test_X), test_Y)

    return accuracy

def run (folder, paramsFolder, output):
    testFolder = folder.replace("_Train", "_Test")
    files = [f for f in listdir(folder) if isfile(join(folder, f))]
    files = sorted(files)

    svm_acc = []

    max_values_svm = []
    index_set_svm = []
    mean_value_svm = []
    std_value_svm = []

    datasetNames = []

    for file in files:
        if file.endswith(".arff"):
            svm_acc.append([])

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

            for paramsSet in range(0, len(params), 4):
                c = float(params[paramsSet])
                kernel = getKernel(int(params[paramsSet + 1]))
                degree = int(params[paramsSet + 2])
                gamma = float(params[paramsSet + 3])
                svm_acc[len(svm_acc) - 1].append(run_svm(train_X, train_Y, test_X, test_Y, c, kernel, degree, gamma))

            max_values_svm.append(np.nanmax(svm_acc[len(svm_acc) - 1]))
            index_set_svm.append(np.nanargmax(svm_acc[len(svm_acc) - 1]))

            mean_value_svm.append(np.nanmean(svm_acc[len(svm_acc) - 1]))
            std_value_svm.append(np.nanstd(svm_acc[len(svm_acc) - 1], ddof=1))

            datasetNames.append(file[:-5])

            outputText = "{0}\nSVM: {1}({2})[{3}]\n\n".format(file, np.mean(svm_acc[len(svm_acc) - 1]), np.std(svm_acc[len(svm_acc) - 1], ddof=1), np.argmax(svm_acc[len(svm_acc) - 1]))
            print outputText

    writeResults(output, "svm", svm_acc, max_values_svm, index_set_svm, mean_value_svm, std_value_svm, datasetNames)

def writeResults(outputPath, method, accs, max_values, index_set, mean_value, std_value, datasetNames):

    outputFile = open(join(outputPath, "{0}-l100.csv".format(method)), 'w+')

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
        return 'linear'
    elif kernel == 2:
        return 'poly'
    elif kernel == 3:
        return 'rbf'
    else:
        return 'sigmoid'

parser = argparse.ArgumentParser()
parser.add_argument('-i', help='Train Data Directory', required=True)
parser.add_argument('-p', help='Parameters', required=True)
parser.add_argument('-o', help='Output', required=True)
args = parser.parse_args()

folder = args.i
params = args.p
output = args.o

if not os.path.isdir(output): os.mkdir(output)

run(folder, params, output)