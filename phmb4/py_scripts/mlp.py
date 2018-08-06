import argparse
from sklearn import neural_network
from sklearn import metrics
from sklearn import preprocessing
import pandas as pd
import numpy as np
from scipy.io import arff
import os
from os import listdir
from os.path import isfile, join

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

def run (folder, paramsFolder, output, supervision):
    testFolder = folder.replace("_Train", "_Test")
    files = [f for f in listdir(folder) if isfile(join(folder, f))]
    files = sorted(files)

    mlp_acc = []

    max_values_mlp = []
    index_set_mlp = []
    mean_value_mlp = []
    std_value_mlp = []

    datasetNames = []

    for file in files:
        if file.endswith(".arff") and "sup_" in file:
            mlp_acc.append([])

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

            for paramsSet in range(0, len(params), 8):
                neurons = int(params[paramsSet])
                hidden_layers = int(params[paramsSet + 1])
                lr = float(params[paramsSet + 2])
                momentum = float(params[paramsSet + 3])
                mlp_epochs = int(params[paramsSet + 4])
                activation = getActivation(int(params[paramsSet + 5]))
                lr_decay = getDecay(int(params[paramsSet + 6]))
                solver = getSolver(int(params[paramsSet + 7]))
                mlp_acc[len(mlp_acc) - 1].append(run_mlp(train_X, train_Y, test_X, test_Y, neurons, hidden_layers, lr,
                                       momentum, mlp_epochs, activation, lr_decay, solver))

            max_values_mlp.append(np.nanmax(mlp_acc[len(mlp_acc) - 1]))
            index_set_mlp.append(np.nanargmax(mlp_acc[len(mlp_acc) - 1]))

            mean_value_mlp.append(np.nanmean(mlp_acc[len(mlp_acc) - 1]))
            std_value_mlp.append(np.nanstd(mlp_acc[len(mlp_acc) - 1], ddof=1))

            datasetNames.append(file[:-5])

            outputText = "{0}\nMLP: {1}({2})[{3}]\n\n".format(file, np.mean(mlp_acc[len(mlp_acc) - 1]), np.std(mlp_acc[len(mlp_acc) - 1], ddof=1), np.argmax(mlp_acc[len(mlp_acc) - 1]))
            print outputText

    writeResults(output, supervision, "mlp", mlp_acc, max_values_mlp, index_set_mlp, mean_value_mlp,
                 std_value_mlp, datasetNames)

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
parser.add_argument('-o', help='Output', required=True)
parser.add_argument('-s', help='Percentage of Supervision', required=True, type=float)
args = parser.parse_args()

folder = args.i
params = args.p
output = args.o
supervision = args.s

if not os.path.isdir(output): os.mkdir(output)

run(folder, params, output, supervision)