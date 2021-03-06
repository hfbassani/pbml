import pandas as pd
import numpy as np
from os.path import join
import os
import sys
import re
from scipy.io import arff


def save_params_file(results, starting_param_name, filename):
    parameters = results.rename(columns=results.iloc[0])
    parameters = parameters.drop([0])
    parameters = parameters.astype(np.float64)
    parameters = parameters.iloc[:, parameters.columns.get_loc(starting_param_name):]

    min_row = parameters.min(0)
    max_row = parameters.max(0)
    min_max = pd.DataFrame([list(min_row), list(max_row)], columns=parameters.columns)

    full_data = min_max.append(parameters, ignore_index=True)

    first_column = map(str, range(len(parameters.index)))
    first_column.insert(0, 'max')
    first_column.insert(0, 'min')
    full_data.insert(0, '', first_column)

    if filename.endswith("/"):
        filename = filename[:-1]

    full_data.to_csv(join(filename, "parameters-" + filename + ".csv"), sep=',', index=False)


def read_header(files, folder, header_rows, save_parameters=True):
    datasets = []
    folds = []
    headers = []

    for file in files:
        if ".csv" in file:
            header = pd.read_csv(join(folder, file), nrows=header_rows, header=None)
            header = header.transpose()
            header = header.rename(columns=header.iloc[0])
            header = header.drop([0])
            header = header.dropna(axis=0, how='any')
            header = header.astype(np.float64)

            headers.append(header)

            if len(datasets) <= 0:
                results = pd.read_csv(join(folder, file), skiprows=header_rows + 1, header=None)

                datasets = results.iloc[0]
                if 'a_t' in datasets.values:
                    datasets = datasets[1: datasets[datasets == "a_t"].index[0]]
                    if save_parameters:
                        save_params_file(results, "a_t", folder)
                elif 'nnodes' in datasets.values:
                    datasets = datasets[1: datasets[datasets == "nnodes"].index[0]]
                    if save_parameters:
                        save_params_file(results, "nnodes", folder)
                elif 'lp' in datasets.values:
                    datasets = datasets[1: datasets[datasets == "lp"].index[0]]
                    if save_parameters:
                        save_params_file(results, "lp", folder)
                else:
                    datasets = datasets[1:]

                pattern = re.compile("(_x\d_k\d)")
                folds = map(lambda x: x if pattern.search(x) is None else x[pattern.search(x).start(0) - 1:],
                            datasets)
                datasets = np.unique(map(lambda x: x if pattern.search(x) is None else x[:pattern.search(x).start(0)],
                                         datasets))

    return datasets, folds, headers


def get_params_and_results(file_name, rows=9):
    results = pd.read_csv(file_name, skiprows=rows + 1, header=None)

    first_param_idx = results.iloc[0]

    if 'a_t' in first_param_idx.values:
        first_param_idx = first_param_idx[first_param_idx == "a_t"].index[0]
    elif 'nnodes' in first_param_idx.values:
        first_param_idx = first_param_idx[first_param_idx == "nnodes"].index[0]
    elif 'lp' in first_param_idx.values:
        first_param_idx = first_param_idx[first_param_idx == "lp"].index[0]
    else:
        first_param_idx = None

    if first_param_idx is not None:
        params = results.drop(results.columns[range(first_param_idx)], axis=1)
        params = params.rename(columns=params.iloc[0])
        params = params.drop([0])
        params = params.astype(np.float64)

        results = results.drop(results.columns[range(first_param_idx, len(results.columns))], axis=1)
        results = results.drop(results.columns[0], axis=1)
        results = results.rename(columns=results.iloc[0])
        results = results.drop([0])
    else:
        params = None
        results = None

    return params, results


def get_data_targets(path, file, target_idx=None):

    if file.endswith(".arff"):
        data, _ = arff.loadarff(open(join(path, file), 'rb'))
        targets = data['class'] if target_idx is None else data[target_idx]
    else:
        data = pd.read_csv(join(path, file), header=None)
        targets = data.iloc[:, -1].values.astype('int16') if target_idx is None else data.ix[:, target_idx].values.astype('int16')

    return targets


def create_folders(path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))


def check_directory(file_path):
    checked_file_path = file_path

    if os.path.isdir(file_path):
        if not file_path.endswith("/"):
            checked_file_path += "/"
    else:
        sys.exit("Invalid directory")

    return checked_file_path


def get_type(type):
    if type == "numeric":
        return "real"


def write_arff_file(arff_file_name, data, meta, output_path, labels):
    new_file = open(join(output_path, arff_file_name), 'w+')
    new_file.write("@relation {0}\n".format(meta.name))
    for i in xrange(len(meta.names())):
        attr = meta.names()[i]
        if attr != "class":
            new_file.write("@attribute {0} {1}\n".format(attr, get_type(meta.types()[i])))
        else:
            new_file.write("@attribute {0} {{".format(attr))
            sorted_labels = sorted(map(int, labels))
            new_file.write("{0}".format(",".join(map(str, sorted_labels))))
            new_file.write("}\n")
    new_file.write("@data\n")
    for _, row in data.iterrows():
        new_file.write(",".join(map(str, row)) + "\n")


def write_general_results(output_path, supervision, method, accs, max_values, index_set, mean_value, std_value, dataset_names):
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
