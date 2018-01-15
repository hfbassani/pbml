import argparse
import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join

def analyse (folder, rows):
    headerRows = rows
    if rows == None:
        headerRows = 2

    files = [f for f in listdir(folder) if isfile(join(folder, f))]
    files = sorted(files)

    datasets = []
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
                results = pd.read_csv(join(folder, file), skiprows=headerRows, header=None)

                datasets = results.iloc[0]
                firstParamIndex = datasets[datasets == "a_t"].index[0]
                datasets = datasets[1: firstParamIndex]

                params = results.drop(results.columns[range(firstParamIndex)], axis=1)
                params = params.rename(columns=params.iloc[0])
                params = params.drop([0])
                params = params.astype(np.float64)
                params = params.reset_index(drop=True)

                results = results.drop(results.columns[range(firstParamIndex, len(results.columns))], axis=1)
                results = results.drop(results.columns[0], axis=1)
                results = results.rename(columns=results.iloc[0])
                results = results.drop([0])
        else:
            file

    index_sets = []
    line = " \t" + "\t".join(datasets) + "\n"
    for i in range(len(headers)):
        local_max_mean = headers[i]["max_mean"]
        index_sets.append(np.array(headers[i]["index_set"]))
        local_num_index_set = headers[i]["index_set"]

        line += files[i] + "\n"
        line += "means\t" + "\t".join(map(str, local_max_mean)) + "\n"
        line += "index_set\t" + "\t".join(map(str, local_num_index_set)) + "\n"
        # line += "mean_value\t" + "\t".join(map(str, means_mean_value)) + "\n\n"
        # line += "\t" + "\t".join(map(str, std_max_values)) + "\n"

    datasets_arr = np.array(datasets)
    index_sets = np.array(index_sets).transpose()
    for i in range(len(datasets_arr)):
        line += datasets_arr[i] + "\n"
        unique_indexes = map(int, np.unique(index_sets[i]))
        for index in unique_indexes:
            line += str(index) + "," + ",".join(map(str, params.iloc[index])) + "\n"

        line += "\n"

    outputFile = open(join(folder, "analysis.csv"), "w+")
    outputFile.write(line)

parser = argparse.ArgumentParser()
parser.add_argument('-i', help='Directory Path', required=True)
parser.add_argument('-r', help='Number of Heade Rows', required=False, type=int)
args = parser.parse_args()

folder = args.i
rows = args.r

analyse(folder, rows)