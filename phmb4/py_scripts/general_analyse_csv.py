import argparse
import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join

def analyse (folder, rows):
    headerRows = rows
    if rows == None:
        headerRows = 4

    files = [f for f in listdir(folder) if isfile(join(folder, f)) and not f.startswith('.') and f.endswith(".csv") and not f.startswith('analysis-')]
    files = sorted(files, key=lambda x: int(x[:-4].split("-l")[1]))

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
                elif 'num_nodes' in datasets.values:
                    datasets = datasets[1: datasets[datasets == "num_nodes"].index[0]]
                else:
                    datasets = datasets[1:]

                folds = list(map(lambda x:x[len(x) - 5:],datasets))
                datasets = np.unique(map(lambda x: x.split("_x")[0], datasets))

    line = " \t" + "\t".join(datasets) + "\n"
    for i in range(len(headers)):
        local_max_values = headers[i]["max_value"]

        datasets_max_values = []

        means_max_values = []
        std_max_values = []

        for j in range(0, len(folds), len(folds) / len(datasets)):
            local_data = np.array(local_max_values)[j :j + len(folds) / len(datasets)]
            datasets_max_values.append(local_data)
            means_max_values.append(np.nanmean(local_data))
            std_max_values.append(np.nanstd(local_data, ddof=1))

        line += files[i] + "\n"
        line += "means_max_values\t" + "\t".join(map(str, means_max_values)) + "\n"
        line += "std_max_values\t" + "\t".join(map(str, std_max_values)) + "\n"

    outputFile = open(join(folder, "analysis-" + folder + ".csv"), "w+")
    outputFile.write(line)

parser = argparse.ArgumentParser()
parser.add_argument('-i', help='Directory Path', required=True)
parser.add_argument('-r', help='Number of Heade Rows', required=False, type=int)
args = parser.parse_args()

folder = args.i
rows = args.r

analyse(folder, rows)