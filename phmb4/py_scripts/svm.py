import argparse
from sklearn import svm
from sklearn import metrics
import pandas as pd
import numpy as np
from scipy.io import arff
import os
from os import listdir
from os.path import isfile, join
import utils


def run_svm(train_x, train_y, test_x, test_y, c=1.0, kernel='rbf', degree=3, gamma=0.1, coef0=0.0):
    clf = svm.SVC(C=c, kernel=kernel, degree=degree, gamma=gamma,
                  coef0=coef0, shrinking=True, probability=False,
                  tol=1e-3, cache_size=200, class_weight=None,
                  verbose=False, max_iter=-1, decision_function_shape=None,
                  random_state=None)

    clf.fit(train_x, train_y)

    accuracy = metrics.accuracy_score(clf.predict(test_x), test_y)

    return accuracy


def run(folder, params_folder, output, supervision, sup_prefix):
    test_folder = folder.replace("_Train", "_Test")

    files = [f for f in listdir(folder) if isfile(join(folder, f)) and f.endswith(".arff") and not f.startswith(".")]

    if sup_prefix and supervision != 1.0:
        files = [f for f in files if f.startswith("sup_")]
    elif not sup_prefix:
        files = [f for f in files if not f.startswith("sup_")]

    files = sorted(files)

    svm_acc = []

    max_values_svm = []
    index_set_svm = []
    mean_value_svm = []
    std_value_svm = []

    dataset_names = []

    for file in files:
        if file.endswith(".arff"):
            svm_acc.append([])

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

            for paramsSet in range(0, len(params), 5):
                c = float(params[paramsSet])
                kernel = get_kernel(int(params[paramsSet + 1]))
                degree = int(params[paramsSet + 2])
                gamma = float(params[paramsSet + 3])
                coef0 = float(params[paramsSet + 4])
                svm_acc[len(svm_acc) - 1].append(run_svm(train_x, train_y, test_x, test_y,
                                                         c, kernel, degree, gamma, coef0))

            max_values_svm.append(np.max(svm_acc[len(svm_acc) - 1]))
            index_set_svm.append(np.argmax(svm_acc[len(svm_acc) - 1]))

            mean_value_svm.append(np.mean(svm_acc[len(svm_acc) - 1]))
            std_value_svm.append(np.std(svm_acc[len(svm_acc) - 1], ddof=1))

            dataset_names.append(file[:-5])

            print "{0}\nSVM: {1}({2})[{3}]\n\n".format(file,
                                                       np.mean(svm_acc[len(svm_acc) - 1]),
                                                       np.std(svm_acc[len(svm_acc) - 1], ddof=1),
                                                       np.argmax(svm_acc[len(svm_acc) - 1]))

    utils.write_general_results(output, supervision, "svm", svm_acc, max_values_svm, index_set_svm,
                                mean_value_svm, std_value_svm, dataset_names)


def get_kernel(kernel):
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
parser.add_argument('-s', help='Percentage of Supervision', required=True, type=float)
parser.add_argument('--sup', help='Sup_ prefix', action='store_false', required=False)
args = parser.parse_args()

folder = args.i
params = args.p
output = args.o
supervision = args.s
sup_prefix = args.sup

if not os.path.isdir(output):
    os.mkdir(output)

run(folder, params, output, supervision, sup_prefix)
