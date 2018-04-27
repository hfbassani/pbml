import pandas as pd
import numpy as np
from os.path import join
import os
import sys

def save_params_file(results, starting_param_name, fileName):
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

    full_data.to_csv(join(fileName, "parameters-" + fileName + ".csv"), sep=',', index=False)

def read_header(files, folder, headerRows):
    datasets = []
    folds = []
    headers = []

    for file in files:
        if ".csv" in file:
            header = pd.read_csv(join(folder, file), nrows=headerRows, header=None)
            header = header.transpose()
            header = header.rename(columns=header.iloc[0])
            header = header.drop([0])
            header = header.dropna(axis=0, how='any')
            header = header.astype(np.float64)

            headers.append(header)

            if len(datasets) <= 0:
                results = pd.read_csv(join(folder, file), skiprows=headerRows + 1, header=None)

                datasets = results.iloc[0]
                if 'a_t' in datasets.values:
                    datasets = datasets[1: datasets[datasets == "a_t"].index[0]]
                    save_params_file(results, "a_t", folder)
                elif 'nnodes' in datasets.values:
                    datasets = datasets[1: datasets[datasets == "nnodes"].index[0]]
                    save_params_file(results, "nnodes", folder)
                else:
                    datasets = datasets[1:]

                folds = list(map(lambda x:x[len(x) - 5:],datasets))
                datasets = np.unique(map(lambda x: x.split("_x")[0], datasets))

    return datasets, folds, headers

def createFolders (path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

def check_directory(filePath):
    if os.path.isdir(filePath):
        if not filePath.endswith("/"):
            return filePath + "/"
    else:
        sys.exit("Invalid directory")

def get_type(type):
    if type == "numeric":
        return "real"