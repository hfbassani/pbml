import argparse
import pandas as pd
import numpy as np
from sklearn import metrics
from os import listdir
from os.path import isfile, join
import utils


def eval(results_paths, true_path, r, output_path, param_file=None, param_names_file=None):
    accs = []
    max_values = []
    max_value_num_nodes = []
    max_value_num_noisy = []
    max_value_num_unlabeled_samples = []
    max_value_num_correct_samples = []
    max_value_num_incorrect_samples = []
    index_set = []
    mean_value = []
    std_value = []
    datasets_names = []

    num_nodes = []
    noisy_samples = []
    unlabeled_samples = []
    incorrect_samples = []
    correct_samples = []

    files = [f for f in listdir(true_path) if isfile(join(true_path, f)) and not f.endswith(".true")]
    files = sorted(files)

    for file in files:
        data = utils.get_data_targets(true_path, file)

        accs.append([])
        num_nodes.append([])
        noisy_samples.append([])
        unlabeled_samples.append([])
        incorrect_samples.append([])
        correct_samples.append([])

        for i in range(r):
            results = open(join(results_paths, "{0}_{1}.results".format(file.split(".")[0], i)), 'rb')
            results = results.readlines()
            print join(results_paths, "{0}_{1}.results".format(file.split(".")[0], i))
            nodes = (int)(results[0].split("\t")[0])
            num_nodes[len(num_nodes) - 1].append(nodes)

            if nodes + 1 < len(results):
                results = pd.read_csv(join(results_paths, "{0}_{1}.results".format(file.split(".")[0], i)),
                                      sep="\t", skiprows=nodes + 1, header=None)

                noisy_samples[len(noisy_samples) - 1].append(len(data) - len(results[len(results.columns) - 1]))

                # PAY ATTENTION
                unlabeled_samples[len(unlabeled_samples) - 1].append(
                    len(results.ix[results[len(results.columns) - 1] == 999]))

                # results = results.ix[results[len(results.columns) - 1] != 999]

                indexes = np.array(results.iloc[:, 0])
                predict = np.array(results.iloc[:, 2])
                true = map(int, data[indexes])

                corrects = metrics.accuracy_score(predict, true, normalize=False)
                correct_samples[len(correct_samples) - 1].append(corrects)
                accuracy = float(corrects) / len(data)
                accs[len(accs) - 1].append(accuracy)

                results_inc = results.ix[results[len(results.columns) - 1] != 999]
                predict_inc = np.array(results_inc.iloc[:, 2])
                incorrects = len(predict_inc) - metrics.accuracy_score(predict, true, normalize=False)

                incorrect_samples[len(incorrect_samples) - 1].append(incorrects)
            else:
                noisy_samples[len(noisy_samples) - 1].append(len(data))
                unlabeled_samples[len(unlabeled_samples) - 1].append(0)
                accs[len(accs) - 1].append(0)
                incorrect_samples[len(incorrect_samples) - 1].append(0)
                correct_samples[len(correct_samples) - 1].append(0)

        max_value_index = np.nanargmax(accs[len(accs) - 1])
        max_values.append(np.nanmax(accs[len(accs) - 1]))
        index_set.append(np.nanargmax(accs[len(accs) - 1]))

        max_value_num_nodes.append(num_nodes[len(num_nodes) - 1][max_value_index])
        max_value_num_noisy.append(noisy_samples[len(noisy_samples) - 1][max_value_index])
        max_value_num_unlabeled_samples.append(unlabeled_samples[len(unlabeled_samples) - 1][max_value_index])
        max_value_num_correct_samples.append(correct_samples[len(correct_samples) - 1][max_value_index])
        max_value_num_incorrect_samples.append(incorrect_samples[len(incorrect_samples) - 1][max_value_index])

        mean_value.append(np.nanmean(accs[len(accs) - 1]))
        std_value.append(np.nanstd(accs[len(accs) - 1], ddof=1))
        datasets_names.append(file.split(".")[0])

    outputFile = open(output_path + '.csv', 'w+')

    line = "max_value," + ",".join(map(str, max_values)) + "\n"
    line += "index_set," + ",".join(map(str, index_set)) + "\n"
    line += "num_nodes," + ",".join(map(str, max_value_num_nodes)) + "\n"
    line += "num_noisy_samples," + ",".join(map(str, max_value_num_noisy)) + "\n"
    line += "num_unlabeled_samples," + ",".join(map(str, max_value_num_unlabeled_samples)) + "\n"
    line += "num_correct_samples," + ",".join(map(str, max_value_num_correct_samples)) + "\n"
    line += "num_incorrect_samples," + ",".join(map(str, max_value_num_incorrect_samples)) + "\n"
    line += "mean_value," + ",".join(map(str, mean_value)) + "\n"
    line += "std_value," + ",".join(map(str, std_value)) + "\n\n"

    if param_file != None and param_names_file != None:
        params = open(param_file, 'r')
        params = params.readlines()
        params = map(float, params)

        names = open(param_names_file, 'r')
        names = names.readlines()
        names = list(map(lambda x: x.strip(), names))

        line += "experiment," + ",".join(datasets_names) + "," + ",".join(names) + "\n"

        for i in range(len(accs[0])):
            line += str(i)
            for j in range(len(datasets_names)):
                line += "," + str(accs[j][i])

            for k in range(len(names)):
                line += "," + str(params[i * len(names) + k])
            line += "\n"
    else:
        line += "experiment," + ",".join(datasets_names) + "\n"

        for i in range(len(accs[0])):
            line += str(i)
            for j in range(len(datasets_names)):
                line += "," + str(accs[j][i])
            line += "\n"

    outputFile.write(line)


parser = argparse.ArgumentParser()
parser.add_argument('-t', help='True Directory', required=True)
parser.add_argument('-i', help='Results Directory', required=True)
parser.add_argument('-r', help='Repeat', required=True, type=int)
parser.add_argument('-n', help='Parameter Names', required=False)
parser.add_argument('-p', help='Parameters', required=False)
parser.add_argument('-o', help='Output', required=True)
args = parser.parse_args()

true = args.t
results = args.i
repeat = args.r
output = args.o
params = args.p
paramNames = args.n

utils.create_folders(output)
eval(results_paths=results, true_path=true, r=repeat, output_path=output, param_file=params, param_names_file=paramNames)
