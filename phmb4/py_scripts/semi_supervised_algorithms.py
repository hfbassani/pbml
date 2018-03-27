import argparse
from sklearn import semi_supervised
from sklearn import metrics
from sklearn import preprocessing
import pandas as pd
import numpy as np
from scipy.io import arff
from os import listdir
from os.path import isfile, join
import random

def run_LabelPropagation(train_X, train_Y, test_X, test_Y, kernel='rbf', gamma=20, n_neighbors=7, max_iter=1000):

    clf = semi_supervised.LabelPropagation(kernel=kernel, gamma=gamma, n_neighbors=n_neighbors, max_iter=max_iter)

    clf.fit(train_X, train_Y)

    accuracy = metrics.accuracy_score(clf.predict(test_X), test_Y)

    return accuracy

def run_LabelSpreading(train_X, train_Y, test_X, test_Y, kernel='rbf', gamma=20, n_neighbors=7, alpha=0.2, max_iter=30):

    clf = semi_supervised.LabelSpreading(kernel=kernel, gamma=gamma, n_neighbors=n_neighbors, alpha=alpha, max_iter=max_iter)

    clf.fit(train_X, train_Y)

    accuracy = metrics.accuracy_score(clf.predict(test_X), test_Y)

    return accuracy

def todo (folder, paramsFolder, numDatasets, output, supervision):
    testFolder = folder.replace("_Train", "_Test")
    files = [f for f in listdir(folder) if isfile(join(folder, f))]
    files = sorted(files)

    if supervision == 1.0:
        outputFile = open("semi{0}-l100.results".format(output), 'w+')
    else:
        outputFile = open("semi{0}-l{1}.results".format(output, ('%.2f' % (supervision)).split(".")[1]), 'w+')

    arffFiles = []
    spreading_acc = []
    propagation_acc = []
    for file in files:
        if ".arff" in file:
            arffFiles.append(file)
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
                propagation_acc[len(propagation_acc) - 1].append(run_LabelPropagation(train_X, labels_prop, test_X, test_Y,
                                                                                      kernel=kernel_propagation,
                                                                                      gamma=gamma_propagation,
                                                                                      n_neighbors=n_neighbors_propagation,
                                                                                      max_iter=max_iter_propagation))

            outputText = "{0}\nSpreading: {1}({2})[{3}]\nPropagation: {4}({5})[{6}]\n\n".format(file,
                                                                                                np.mean(spreading_acc[len(spreading_acc) - 1]), np.std(spreading_acc[len(spreading_acc) - 1], ddof=1), np.argmax(spreading_acc[len(spreading_acc) - 1]),
                                                                                                np.mean(propagation_acc[len(propagation_acc) - 1]), np.std(propagation_acc[len(propagation_acc) - 1], ddof=1), np.argmax(propagation_acc[len(propagation_acc) - 1]))
            print outputText
            outputFile.write(outputText)

    writeMeans(spreading_acc, numDatasets, arffFiles, outputFile, "Spreading")
    writeMeans(propagation_acc, numDatasets, arffFiles, outputFile, "Propagation")

    writeBests(propagation_acc, numDatasets, arffFiles, outputFile, "Propagation")
    writeBests(spreading_acc, numDatasets, arffFiles, outputFile, "Spreading")

def writeMeans(accs, numDatasets, arffFiles, outputFile, title):
    nRow = len(accs) / numDatasets
    outputFile.write("----> {0} Means (stds)\n".format(title))
    for i in range(nRow):
        outputFile.write(arffFiles[i][len(arffFiles[i]) - 10:-5] + '\t\t')
        for j in range(0, len(accs), nRow):
            outputFile.write("{0:.4f} ({1:.4f})\t".format(np.mean(accs[j + i]), np.std(accs[j + i], ddof=1)))
        outputFile.write("\n")

    outputFile.write("Mean (std)\t")
    for i in range(0, len(accs), nRow):
        m_means = []
        for j in range(nRow):
            m_means.append(np.mean(accs[j + i]))
        outputFile.write("{0:.4f} ({1:.4f})\t".format(np.mean(m_means), np.std(m_means, ddof=1)))

    outputFile.write("\n\n")

def writeBests(accs, numDatasets, arffFiles, outputFile, title):
    nRow = len(accs) / numDatasets
    outputFile.write("----> {0} Bests\n".format(title))
    for i in range(nRow):
        outputFile.write(arffFiles[i][len(arffFiles[i]) - 10:-5] + '\t\t')
        for j in range(0, len(accs), nRow):
            outputFile.write("{0:.4f}\t\t\t".format(np.amax(accs[j + i])))
        outputFile.write("\n")

    outputFile.write("Mean (std)\t")
    for i in range(0, len(accs), nRow):
        m_means = []
        for j in range(nRow):
            m_means.append(np.amax(accs[j + i]))
        outputFile.write("{0:.4f} ({1:.4f})\t".format(np.mean(m_means), np.std(m_means, ddof=1)))

    outputFile.write("\n\n")

def getKernel(kernel):
    if kernel == 1:
        return 'rbf'
    else:
        return 'knn'

parser = argparse.ArgumentParser()
parser.add_argument('-i', help='Train Data Directory', required=True)
parser.add_argument('-p', help='Parameters', required=True)
parser.add_argument('-n', help='Number of Datasets', required=True, type=int)
parser.add_argument('-o', help='Output', required=True)
parser.add_argument('-s', help='Percentage of Supervision', required=True, type=float)
args = parser.parse_args()

folder = args.i
params = args.p
n = args.n
output = args.o
supervision = args.s

todo(folder, params, n, output, supervision)