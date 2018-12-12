import argparse
from sklearn import neural_network
from sklearn import metrics
import pandas as pd
import numpy as np
from scipy.io import arff
import os
from os import listdir
from os.path import isfile, join


def run_mlp(train_x, train_y, test_x, test_y, neurons=100, hidden_layers=1, lr=0.001, momentum=0.9,
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

    mlp_acc = []

    max_values_mlp = []
    index_set_mlp = []
    mean_value_mlp = []
    std_value_mlp = []

    dataset_names = []

    for file in files:
        mlp_acc.append([])

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

        alldata = np.append(train_x, test_x, axis=0)

        train_x = alldata[:len(train_x)]
        test_x = alldata[len(train_x):]

        params = open(params_folder, 'r')
        params = np.array(params.readlines())

        for paramsSet in range(0, len(params), 8):
            neurons = int(params[paramsSet])
            hidden_layers = int(params[paramsSet + 1])
            lr = float(params[paramsSet + 2])
            momentum = float(params[paramsSet + 3])
            mlp_epochs = int(params[paramsSet + 4])
            activation = get_activation(int(params[paramsSet + 5]))
            lr_decay = get_decay(int(params[paramsSet + 6]))
            solver = get_solver(int(params[paramsSet + 7]))
            mlp_acc[len(mlp_acc) - 1].append(run_mlp(train_x, train_y, test_x, test_y, neurons, hidden_layers, lr,
                                                     momentum, mlp_epochs, activation, lr_decay, solver))

        max_values_mlp.append(np.nanmax(mlp_acc[len(mlp_acc) - 1]))
        index_set_mlp.append(np.nanargmax(mlp_acc[len(mlp_acc) - 1]))

        mean_value_mlp.append(np.nanmean(mlp_acc[len(mlp_acc) - 1]))
        std_value_mlp.append(np.nanstd(mlp_acc[len(mlp_acc) - 1], ddof=1))

        dataset_names.append(test_file[:-5])

        print "{0}\nMLP: {1}({2})[{3}]\n\n".format(file,
                                                   np.mean(mlp_acc[len(mlp_acc) - 1]),
                                                   np.std(mlp_acc[len(mlp_acc) - 1], ddof=1),
                                                   np.argmax(mlp_acc[len(mlp_acc) - 1]))

    write_results(output, supervision, "mlp", mlp_acc, max_values_mlp, index_set_mlp, mean_value_mlp,
                  std_value_mlp, dataset_names)


def write_results(output_path, supervision, method, accs, max_values, index_set, mean_value, std_value, dataset_names):
    if supervision == 1.0:
        output_file = open(join(output_path, "{0}-l100.csv".format(method)), 'w+')
    else:
        output_file = open(
            join(output_path, "{0}-l{1}.csv".format(method, ('%.2f' % supervision).split(".")[1])), 'w+')

    line = "max_value," + ",".join(map(str, max_values)) + "\n"
    line += "index_set," + ",".join(map(str, index_set)) + "\n"
    line += "mean_value," + ",".join(map(str, mean_value)) + "\n"
    line += "std_value," + ",".join(map(str, std_value)) + "\n\n"

    line += "experiment," + ",".join(dataset_names) + "\n"

    for i in range(len(accs[0])):
        line += str(i)
        for j in range(len(dataset_names)):
            line += "," + str(accs[j][i])
        line += "\n"

    output_file.write(line)


def get_activation(activation):
    if activation == 1:
        return 'logistic'
    elif activation == 2:
        return 'tanh'
    else:
        return 'relu'


def get_decay(lr_decay):
    if lr_decay == 1:
        return 'constant'
    elif lr_decay == 2:
        return 'invscaling'
    else:
        return 'adaptive'


def get_solver(solver):
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
