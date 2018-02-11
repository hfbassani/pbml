import argparse
import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn import metrics
from os import listdir
from os.path import isfile, join
import os

def eval (resultsPath, truePath, r, outputPath, paramFile, paramNamesFile):

    accs = []
    max_values = []
    index_set = []
    mean_value = []
    std_value = []
    datasetNames = []

    files = [f for f in listdir(truePath) if isfile(join(truePath, f))]
    files = sorted(files)

    for file in files:
        if file.endswith(".arff"):
            data, _ = arff.loadarff(open(join(truePath, file), 'rb'))
            data = data['class']
            accs.append([])
            for i in range(r):
                results = open(join(resultsPath, "{0}_{1}.results".format(file[:-5], i)), 'rb')
                results = results.readlines()
                print join(resultsPath, "{0}_{1}.results".format(file[:-5], i))
                skipRows = (int)(results[0].split("\t")[0])

                if skipRows + 1 < len(results):
                    results = pd.read_csv(join(resultsPath, "{0}_{1}.results".format(file[:-5], i)),
                                          sep="\t", skiprows=skipRows + 1, header=None)

                    # PAY ATTENTION
                    results = results.ix[results[len(results.columns) - 1] >= 0]

                    indexes = np.array(results.iloc[:,0])
                    predict = np.array(results.iloc[:,2])
                    true = map(int, data[indexes])

                    accs[len(accs) - 1].append(metrics.accuracy_score(predict, true))
                else:
                    accs[len(accs) - 1].append(0)

            max_values.append(np.amax(accs[len(accs) - 1]))
            index_set.append(np.argmax(accs[len(accs) - 1]))
            mean_value.append(np.mean(accs[len(accs) - 1]))
            std_value.append(np.std(accs[len(accs) - 1], ddof=1))
            datasetNames.append(file[:-5])

    num_datasets = 5
    means = []
    for i in range(0, len(accs), len(accs) / num_datasets):
        local_folds_acc = accs[i: i + len(accs) / num_datasets]
        means.append(np.mean(local_folds_acc, axis=0))


    outputFile = open(outputPath + '.csv', 'w+')

    line = "max_mean"
    for i in range(num_datasets):
        line += "," + str(np.nanmax(means[i]))
    line += "\n"

    line += "index_set"
    for i in range(num_datasets):
        line += "," + str(np.nanargmax(means[i]))
    line += "\n"

    params = open(paramFile, 'r')
    params = params.readlines()
    params = map(float, params)

    names = open(paramNamesFile, 'r')
    names = names.readlines()
    names = list(map(lambda x:x.strip(),names))

    line += "experiment," + ",".join(np.unique(map(lambda x: x[:-6], datasetNames))) + "," + ",".join(names) + "\n"

    for i in range(len(means[0])):
        line += str(i)
        for j in range(len(means)):
            line += "," + str(means[j][i])

        for k in range(len(names)):
            line += "," + str(params[i * len(names) + k])
        line += "\n"

    outputFile.write(line)

parser = argparse.ArgumentParser()
parser.add_argument('-t', help='True Directory', required=True)
parser.add_argument('-i', help='Results Directory', required=True)
parser.add_argument('-r', help='Repeat', required=True, type=int)
parser.add_argument('-n', help='Parameter Names', required=True)
parser.add_argument('-p', help='Parameters', required=True)
parser.add_argument('-o', help='Output', required=True)
args = parser.parse_args()

true = args.t
results = args.i
repeat = args.r
output = args.o
params = args.p
paramNames = args.n

eval(resultsPath=results, truePath=true, r=repeat, outputPath=output, paramFile=params, paramNamesFile=paramNames)


