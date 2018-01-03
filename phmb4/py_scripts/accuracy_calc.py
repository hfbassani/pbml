import argparse
import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn import metrics
from os import listdir
from os.path import isfile, join
import os

def eval (resultsPath, truePath, r, outputPath):

    files = [f for f in listdir(truePath) if isfile(join(truePath, f))]
    files = sorted(files)

    output = pd.DataFrame()
    for file in files:
        if file.endswith(".arff"):
            data, _ = arff.loadarff(open(join(truePath, file), 'rb'))
            data = data['class']
            accs = []
            for i in range(r):
                results = open(join(resultsPath, "{0}_{1}.results".format(file[:-5], i)), 'rb')
                results = results.readlines()
                print join(resultsPath, "{0}_{1}.resultsacc".format(file[:-5], i))
                skipRows = (int)(results[0].split("\t")[0])

                results = pd.read_csv(join(resultsPath, "{0}_{1}.results".format(file[:-5], i)),
                                      sep="\t", skiprows=skipRows + 1, header=None)

                # PAY ATTENTION
                results = results.ix[results[len(results.columns) - 1] >= 0]

                indexes = np.array(results.iloc[:,0])
                predict = np.array(results.iloc[:,1])
                true = data[indexes]

                accs.append(metrics.accuracy_score(predict, true))

            accs.append(np.amax(accs))
            accs.append(np.argmax(accs))
            accs.append(np.mean(accs))
            accs.append(np.std(accs, ddof=1))
            output[file[:-5]] = accs

    output.to_csv(outputPath + ".csv", sep="\t", index=False)

parser = argparse.ArgumentParser()
parser.add_argument('-i', help='True Directory', required=True)
parser.add_argument('-t', help='Results Directory', required=True)
parser.add_argument('-r', help='Repeat', required=True, type=int)
parser.add_argument('-o', help='Output', required=True)
args = parser.parse_args()

true = args.i
results = args.t
repeat = args.r
output = args.o

eval(results, true, repeat, output)


