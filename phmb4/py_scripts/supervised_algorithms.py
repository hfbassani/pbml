import argparse
from sklearn import svm
from sklearn import neural_network
from sklearn import metrics
from sklearn import preprocessing
import pandas as pd
import numpy as np
from scipy.io import arff
from os import listdir
from os.path import isfile, join

def run_svm(train_X, train_Y, test_X, test_Y, c=1.0, kernel='rbf', degree=3):
    # clf = svm.SVR()
    clf = svm.SVC(C=c, kernel=kernel, degree=degree, gamma='auto',
                  coef0=0.0, shrinking=True, probability=False,
                  tol=1e-3, cache_size=200, class_weight=None,
                  verbose=False, max_iter=-1, decision_function_shape=None,
                  random_state=None)

    clf.fit(train_X, train_Y)

    accuracy = metrics.accuracy_score(clf.predict(test_X), test_Y)

    return accuracy

def run_mlp(train_X, train_Y, test_X, test_Y, neurons=100, hidden_layers=1, lr=0.001, momentum=0.9,
            mlp_epochs=200, activation='logistic', lr_decay='constant', solver='lbfgs'):
    clf = neural_network.MLPClassifier(hidden_layer_sizes=(neurons,) * hidden_layers, activation=activation,
                                       solver=solver, alpha=0.0001,
                                       batch_size='auto', learning_rate=lr_decay,
                                       learning_rate_init=lr, power_t=0.5, max_iter=mlp_epochs,
                                       shuffle=True, random_state=None, tol=1e-4,
                                       verbose=False, warm_start=False, momentum=momentum,
                                       nesterovs_momentum=True, early_stopping=False,
                                       validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
                                       epsilon=1e-8)

    clf.fit(train_X, train_Y)

    accuracy = metrics.accuracy_score(clf.predict(test_X), test_Y)

    return accuracy

def todo (folder, paramsFolder, numDatasets):
    testFolder = folder.replace("_Train", "_Test")
    files = [f for f in listdir(folder) if isfile(join(folder, f))]
    files = sorted(files)

    outputFile = open("svm_mlp.results", 'w+')

    arffFiles = []
    svm_acc = []
    mlp_acc = []
    for file in files:
        if ".arff" in file:
            arffFiles.append(file)
            svm_acc.append([])
            mlp_acc.append([])

            train_X, meta_trainX = arff.loadarff(open(join(folder, file), 'rb'))
            train_X = pd.DataFrame(train_X)
            train_Y = train_X['class']
            del train_X['class']

            train_X = np.array(train_X)
            train_Y = np.array(train_Y)

            testFile = file.replace("train_", "test_")
            test_X, meta_testX = arff.loadarff(open(join(testFolder, testFile), 'rb'))
            test_X = pd.DataFrame(test_X)
            test_Y = test_X['class']
            del test_X['class']

            test_X = np.array(test_X)
            test_Y = np.array(test_Y)

            alldata = np.append(train_X, test_X, axis=0)
            scaler = preprocessing.MinMaxScaler()
            scaler.fit(alldata)
            alldata = scaler.transform(alldata)

            train_X = alldata[:len(train_X)]
            test_X = alldata[len(train_X):]

            params = open(paramsFolder, 'r')
            params = np.array(params.readlines())

            for paramsSet in range(0, len(params), 11):
                c = float(params[paramsSet])
                kernel = getKernel(int(params[paramsSet + 1]))
                degree = int(params[paramsSet + 2])
                svm_acc[len(svm_acc) - 1].append(run_svm(train_X, train_Y, test_X, test_Y, c, kernel, degree))

                neurons = int(params[paramsSet + 3])
                hidden_layers = int(params[paramsSet + 4])
                lr = float(params[paramsSet + 5])
                momentum = float(params[paramsSet + 6])
                mlp_epochs = int(params[paramsSet + 7])
                activation = getActivation(int(params[paramsSet + 8]))
                lr_decay = getDecay(int(params[paramsSet + 9]))
                solver = getSolver(int(params[paramsSet + 10]))
                mlp_acc[len(mlp_acc) - 1].append(run_mlp(train_X, train_Y, test_X, test_Y, neurons, hidden_layers, lr,
                                       momentum, mlp_epochs, activation, lr_decay, solver))


            outputText = "{0}\nSVM: {1}({2})[{3}]\nMLP: {4}({5})[{6}]\n\n".format(file,
                                                                        np.mean(svm_acc[len(svm_acc) - 1]), np.std(svm_acc[len(svm_acc) - 1], ddof=1), np.argmax(svm_acc[len(svm_acc) - 1]),
                                                                        np.mean(mlp_acc[len(mlp_acc) - 1]), np.std(mlp_acc[len(mlp_acc) - 1], ddof=1), np.argmax(mlp_acc[len(mlp_acc) - 1]))
            print outputText
            outputFile.write(outputText)

    writeMeans(svm_acc, numDatasets, arffFiles, outputFile, "SVM")
    writeBests(svm_acc, numDatasets, arffFiles, outputFile, "SVM")

    writeMeans(mlp_acc, numDatasets, arffFiles, outputFile, "MLP")
    writeBests(mlp_acc, numDatasets, arffFiles, outputFile, "MLP")

def writeMeans (accs, numDatasets, arffFiles, outputFile, title):
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

def writeBests (accs, numDatasets, arffFiles, outputFile, title):
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
        return 'linear'
    elif kernel == 2:
        return 'poly'
    elif kernel == 3:
        return 'rbf'
    else:
        return 'sigmoid'

def getActivation(activation):
    if activation == 1:
        return 'logistic'
    elif activation == 2:
        return 'tanh'
    else:
        return 'relu'

def getDecay(lr_decay):
    if lr_decay == 1:
        return 'constant'
    elif lr_decay == 2:
        return 'invscaling'
    else:
        return 'adaptive'

def getSolver(solver):
    if solver == 1:
        return 'lbfgs'
    elif solver == 2:
        return 'sgd'
    else:
        return 'adam'

parser = argparse.ArgumentParser()
parser.add_argument('-i', help='Train Data Directory', required=True)
parser.add_argument('-p', help='Parameters', required=True)
parser.add_argument('-n', help='Number of Datasets', required=True, type=int)
args = parser.parse_args()

folder = args.i
params = args.p
n = args.n

todo(folder, params, n)