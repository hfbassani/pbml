import argparse
import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join

def csv2latex (folder, rows):
    datasets, method, line, plot_means, plot_stds = summarize(folder, rows)

    latex_table(plot_means, plot_stds, datasets)

    if folder.endswith("/"):
        folder = folder[:-1]

    outputFile = open(join(folder, "analysis-" + folder + ".csv"), "w+")
    outputFile.write(line)

def summarize(folder, rows):
    headerRows = rows
    if rows == None:
        headerRows = 9

    files = [f for f in listdir(folder) if isfile(join(folder, f)) and not f.startswith('.') and f.endswith(".csv")  and not f.startswith('analysis-')]
    files = sorted(files, key=lambda x: int(x[:-4].split("-l")[-1]))

    method = files[0].split("-l")[0]

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
                    datasets = datasets[1: ]

                folds = list(map(lambda x: x[len(x) - 5:], datasets))
                datasets = np.unique(map(lambda x: x.split("_x")[0], datasets))

    plot_means = []
    plot_stds = []
    line = " \t" + "\t".join(datasets) + "\n"

    for i in range(len(headers)):
        local_max_values = headers[i]["max_value"]

        if rows > 4:
            local_num_nodes = headers[i]["num_nodes"]
            local_num_noisy_samples = headers[i]["num_noisy_samples"]
            local_num_unlabeled_samples = headers[i]["num_unlabeled_samples"]
            local_num_correct_samples = headers[i]["num_correct_samples"]
            local_num_incorrect_samples = headers[i]["num_incorrect_samples"]
        else:
            local_num_nodes = []
            local_num_noisy_samples = []
            local_num_unlabeled_samples = []
            local_num_correct_samples = []
            local_num_incorrect_samples = []

        datasets_max_values = []
        datasets_num_nodes = []
        datasets_num_noisy_samples = []
        datasets_num_unlabeled_samples = []
        datasets_num_correct_samples = []
        datasets_num_incorrect_samples = []

        means_max_values = []
        means_num_nodes = []
        means_num_noisy_samples = []
        means_num_unlabeled_samples = []
        means_num_correct_samples = []
        means_num_incorrect_samples = []
        std_max_values = []

        for j in range(0, len(folds), len(folds) / len(datasets)):
            local_data = np.array(local_max_values)[j:j + len(folds) / len(datasets)]
            datasets_max_values.append(local_data)
            means_max_values.append(np.nanmean(local_data))
            std_max_values.append(np.nanstd(local_data, ddof=1))

            if rows > 4:
                local_nodes = np.array(local_num_nodes)[j:j + len(folds) / len(datasets)]
                datasets_num_nodes.append(local_nodes)
                means_num_nodes.append(np.nanmean(local_nodes))

                local_noisy = np.array(local_num_noisy_samples)[j:j + len(folds) / len(datasets)]
                datasets_num_noisy_samples.append(local_noisy)
                means_num_noisy_samples.append(np.nanmean(local_noisy))

                local_unlabel = np.array(local_num_unlabeled_samples)[j:j + len(folds) / len(datasets)]
                datasets_num_unlabeled_samples.append(local_unlabel)
                means_num_unlabeled_samples.append(np.nanmean(local_unlabel))

                local_correct = np.array(local_num_correct_samples)[j:j + len(folds) / len(datasets)]
                datasets_num_correct_samples.append(local_correct)
                means_num_correct_samples.append(np.nanmean(local_correct))

                local_incorrect = np.array(local_num_incorrect_samples)[j:j + len(folds) / len(datasets)]
                datasets_num_incorrect_samples.append(local_incorrect)
                means_num_incorrect_samples.append(np.nanmean(local_incorrect))

        line += files[i] + "\n"
        line += "means_max_values\t" + "\t".join(map(str, means_max_values)) + "\n"
        line += "std_max_values\t" + "\t".join(map(str, std_max_values)) + "\n"

        if rows > 4:
            line += "means_num_nodes\t" + "\t".join(map(str, means_num_nodes)) + "\n"
            line += "means_num_noisy_samples\t" + "\t".join(map(str, means_num_noisy_samples)) + "\n"
            line += "means_num_unlabeled_samples\t" + "\t".join(map(str, means_num_unlabeled_samples)) + "\n"
            line += "means_correct_samples\t" + "\t".join(map(str, means_num_correct_samples)) + "\n"
            line += "means_incorrect_samples\t" + "\t".join(map(str, means_num_incorrect_samples)) + "\n\n"

        plot_means.append(means_max_values)
        plot_stds.append(std_max_values)

    return datasets, method, line, plot_means, plot_stds

def latex_table(means, stds, datasets):
    table = "\hline \n"
    table += "\\bfseries  Metric & \\bfseries "
    dataset_names = [dataset.replace("sup_", "").split("_")[1].capitalize() for dataset in datasets]

    table += " & \\bfseries ".join(dataset_names) + " \\\\ \n"
    table += "\hline\hline \n"

    for i in xrange(len(means)):
        local_mean = means[i]
        local_std = stds[i]

        for j in xrange(len(local_mean)):
            table += " & " + '%.3f' % (local_mean[j]) + " (" + '%.3f' % (local_std[j]) + ")"

        table += " \\\\ \n"

    table += "\hline"
    print table

parser = argparse.ArgumentParser()
parser.add_argument('-i', help='Directory Path', required=True)
parser.add_argument('-r', help='Number of Heade Rows', required=False, type=int)
args = parser.parse_args()

folder = args.i
rows = args.r

csv2latex(folder, rows)