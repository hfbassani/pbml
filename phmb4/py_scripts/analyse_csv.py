import argparse
import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join

def analyse (folder, rows):
    headerRows = rows
    if rows == None:
        headerRows = 7

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
                results = pd.read_csv(join(folder, file), skiprows=headerRows + 1, header=None)

                datasets = results.iloc[0]#[1:]
                datasets = datasets[1: datasets[datasets == "a_t"].index[0]]

                folds = list(map(lambda x:x[len(x) - 5:],datasets))
                datasets = np.unique(map(lambda x: x[:-6], datasets))

    line = " \t" + "\t".join(datasets) + "\n"
    for i in range(len(headers)):
        local_max_values = headers[i]["max_value"]
        local_num_nodes = headers[i]["num_nodes"]
        local_num_noisy_samples = headers[i]["num_noisy_samples"]
        local_num_unlabeled_samples = headers[i]["num_unlabeled_samples"]
        local_mean_value = headers[i]["mean_value"]

        datasets_max_values = []
        datasets_num_nodes = []
        datasets_num_noisy_samples = []
        datasets_num_unlabeled_samples = []
        datasets_mean_value = []

        means_max_values = []
        means_num_nodes = []
        means_num_noisy_samples = []
        means_num_unlabeled_samples = []
        means_mean_value = []
        std_max_values = []

        for j in range(0, len(folds), len(folds) / len(datasets)):
            local_data = np.array(local_max_values)[j :j + len(folds) / len(datasets)]
            datasets_max_values.append(local_data)
            means_max_values.append(np.nanmean(local_data))
            std_max_values.append(np.nanstd(local_data, ddof=1))

            local_nodes = np.array(local_num_nodes)[j:j + len(folds) / len(datasets)]
            datasets_num_nodes.append(local_nodes)
            means_num_nodes.append(np.nanmean(local_nodes))

            local_noisy = np.array(local_num_noisy_samples)[j:j + len(folds) / len(datasets)]
            datasets_num_noisy_samples.append(local_noisy)
            means_num_noisy_samples.append(np.nanmean(local_noisy))

            local_unlabel = np.array(local_num_unlabeled_samples)[j:j + len(folds) / len(datasets)]
            datasets_num_unlabeled_samples.append(local_unlabel)
            means_num_unlabeled_samples.append(np.nanmean(local_unlabel))

            local_mean_value = np.array(local_mean_value)[j:j + len(folds) / len(datasets)]
            datasets_mean_value.append(local_mean_value)
            means_mean_value.append(np.nanmean(local_mean_value))

        line += files[i] + "\n"
        line += "means_max_values\t" + "\t".join(map(str, means_max_values)) + "\n"
        line += "means_num_nodes\t" + "\t".join(map(str, means_num_nodes)) + "\n"
        line += "means_num_noisy_samples\t" + "\t".join(map(str, means_num_noisy_samples)) + "\n"
        line += "means_num_unlabeled_samples\t" + "\t".join(map(str, means_num_unlabeled_samples)) + "\n\n"
        # line += "mean_value\t" + "\t".join(map(str, means_mean_value)) + "\n\n"
        # line += "\t" + "\t".join(map(str, std_max_values)) + "\n"

    outputFile = open(join(folder + ".csv"), "w+")
    outputFile.write(line)

parser = argparse.ArgumentParser()
parser.add_argument('-i', help='Directory Path', required=True)
parser.add_argument('-r', help='Number of Heade Rows', required=False, type=int)
args = parser.parse_args()

folder = args.i
rows = args.r

analyse(folder, rows)