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
                results = pd.read_csv(join(folder, file), skiprows=headerRows, header=None)

                datasets = results.iloc[0]#[1:]
                datasets = datasets[1: datasets[datasets == "a_t"].index[0]]
        else:
            file

    line = " \t" + "\t".join(datasets) + "\n"
    for i in range(len(headers)):
        local_max_mean = headers[i]["max_mean"]
        local_num_index_set = headers[i]["index_set"]

        line += files[i] + "\n"
        line += "means\t" + "\t".join(map(str, local_max_mean)) + "\n"
        line += "index_set\t" + "\t".join(map(str, local_num_index_set)) + "\n"
        # line += "mean_value\t" + "\t".join(map(str, means_mean_value)) + "\n\n"
        # line += "\t" + "\t".join(map(str, std_max_values)) + "\n"

    outputFile = open(join(folder, "analysis.csv"), "w+")
    outputFile.write(line)

parser = argparse.ArgumentParser()
parser.add_argument('-i', help='Directory Path', required=True)
parser.add_argument('-r', help='Number of Heade Rows', required=False, type=int)
args = parser.parse_args()

folder = args.i
rows = args.r

analyse(folder, rows)