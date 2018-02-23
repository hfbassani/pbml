import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from os import listdir
from os.path import isfile, join

image_path = "plots/"

def analyse (folder, rows, plot, save):
    headerRows = rows
    if rows == None:
        headerRows = 7

    files = [f for f in listdir(folder) if isfile(join(folder, f))]
    files = sorted(files, key = lambda x: int(x[:-4].split("l")[1]))

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
                datasets = np.unique(map(lambda x: x.split("_x")[0], datasets))

    plot_means = []
    plot_stds = []
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
        line += "std_max_values\t" + "\t".join(map(str, std_max_values)) + "\n"
        line += "means_num_nodes\t" + "\t".join(map(str, means_num_nodes)) + "\n"
        line += "means_num_noisy_samples\t" + "\t".join(map(str, means_num_noisy_samples)) + "\n"
        line += "means_num_unlabeled_samples\t" + "\t".join(map(str, means_num_unlabeled_samples)) + "\n\n"
        # line += "mean_value\t" + "\t".join(map(str, means_mean_value)) + "\n\n"

        plot_means.append(means_max_values)
        plot_stds.append(std_max_values)

    plot_graph(plot_means, plot_stds, datasets, plot, save)

    if folder.endswith("/"):
        folder = folder[:-1]

    outputFile = open(join(folder + ".csv"), "w+")
    outputFile.write(line)

def plot_graph(means, stds, datasets, plot, save):
    label_prop_1 = [0.0455, 0.0074, 0.0125, 0.0068, 0.0101, 0.0500, 0.0091]
    label_prop_1err = [0.0107, 0.0050, 0.0085, 0.0058, 0.0015, 0.0130, 0.0055]

    label_spreading_1 = [0.0286, 0.0113, 0.0203, 0.0135, 0.0112, 0.0417, 0.0088]
    label_spreading_1err = [0.0118, 0.0060, 0.0074, 0.0098, 0.0014, 0.0125, 0.0047]

    label_prop_5 = [0.0943, 0.0404, 0.0404, 0.0377, 0.0511, 0.0833, 0.0532]
    label_prop_5err = [0.0197, 0.0186, 0.0353, 0.0254, 0.0037, 0.0283, 0.0212]

    label_spreading_5 = [0.0606, 0.0451, 0.0561, 0.0406, 0.0500, 0.0978, 0.0515]
    label_spreading_5err = [0.0227, 0.0135, 0.0185, 0.0097, 0.0043, 0.0306, 0.0116]

    label_prop_10 = [0.1481, 0.0885, 0.0856, 0.0609, 0.1008, 0.1395, 0.0886]
    label_prop_10err = [0.0291, 0.0212, 0.0305, 0.0144, 0.0091, 0.0384, 0.0205]

    label_spreading_10 = [0.0842, 0.0851, 0.0717, 0.0725, 0.1049, 0.1563, 0.1138]
    label_spreading_10err = [0.0171, 0.0117, 0.0268, 0.0264, 0.0055, 0.0424, 0.0229]

    label_prop_25 = [0.2896, 0.1680, 0.1652, 0.1565, 0.2488, 0.3313, 0.2229]
    label_prop_25err = [0.0451, 0.0346, 0.0493, 0.0361, 0.0127, 0.0508, 0.0167]

    label_spreading_25 = [0.2155, 0.1953, 0.1871, 0.1778, 0.2550, 0.3164, 0.2616]
    label_spreading_25err = [0.0237, 0.0243, 0.0429, 0.0369, 0.0070, 0.0551, 0.0329]

    label_prop_50 = [0.7626, 0.6510, 0.3957, 0.3681, 0.6763, 0.7291, 0.4744]
    label_prop_50err = [0.0312, 0.0266, 0.0510, 0.1176, 0.0436, 0.0737, 0.0297]

    label_spreading_50 = [0.7626, 0.6510, 0.4144, 0.3884, 0.6317, 0.6938, 0.5013]
    label_spreading_50err = [0.0312, 0.0266, 0.0652, 0.0804, 0.0453, 0.0624, 0.0321]

    label_prop_75 = [0.8030, 0.6693, 0.5562, 0.6164, 0.9709, 0.8458, 0.7067]
    label_prop_75err = [0.0283, 0.0253, 0.0495, 0.0341, 0.0051, 0.0751, 0.0472]

    label_spreading_75 = [0.8098, 0.6836, 0.5980, 0.6618, 0.9766, 0.8291, 0.7354]
    label_spreading_75err = [0.0253, 0.0270, 0.0387, 0.0224, 0.0031, 0.0769, 0.0274]

    label_prop_100 = [0.8148, 0.7339, 0.6465, 0.6386, 0.9941, 0.8833, 0.9451]
    label_prop_100err = [0.0271, 0.0326, 0.0613, 0.0450, 0.0013, 0.0585, 0.0183]

    label_spreading_100 = [0.8114, 0.7296, 0.6465, 0.6618, 0.9941, 0.8812, 0.9451]
    label_spreading_100err = [0.0264, 0.0295, 0.0641, 0.0407, 0.0013, 0.0594, 0.0183]

    percentage_values = np.linspace(1, 100, num=7)
    percentage_labels = ['1%', '5%', '10%', '25%', '50%', '75%', '100%']

    plot_means = np.transpose(means)
    plot_stds = np.transpose(stds)

    for i in xrange(len(datasets)):
        # current_values_prop = [label_prop_1[i], label_prop_5[i], label_prop_10[i], label_prop_25[i], label_prop_50[i],
        #                        label_prop_75[i], label_prop_100[i]]
        # current_values_spre = [label_spreading_1[i], label_spreading_5[i], label_spreading_10[i], label_spreading_25[i],
        #                        label_spreading_50[i], label_spreading_75[i], label_spreading_100[i]]
        #
        # current_values_prop_err = [label_prop_1err[i], label_prop_5err[i], label_prop_10err[i], label_prop_25err[i],
        #                            label_prop_50err[i], label_prop_75err[i], label_prop_100err[i]]
        # current_values_spre_err = [label_spreading_1err[i], label_spreading_5err[i], label_spreading_10err[i],
        #                            label_spreading_25err[i], label_spreading_50err[i], label_spreading_75err[i],
        #                            label_spreading_100err[i]]

        fig, ax = plt.subplots()
        ax.yaxis.grid()
        ax.set_ylim([0, 1])
        ax.set_xticklabels(percentage_labels)

        title = datasets[i]
        plt.title(title, fontsize=14)
        plt.yticks(np.linspace(0, 1, num=11))
        plt.xticks(percentage_values)

        plt.errorbar(percentage_values, plot_means[i], plot_stds[i], label='SS-SOM', linestyle='-',
                     marker='o', clip_on=False, markeredgewidth=2, capsize=5)
        # plt.errorbar(percentage_values, current_values_spre, current_values_spre_err, label='Label Spreading',
        #              linestyle='-', marker='D', clip_on=False, markeredgewidth=2, capsize=5)
        # plt.errorbar(percentage_values, current_values_prop, current_values_prop_err, label='Label Propagation',
        #              linestyle='-', marker='x', clip_on=False, markeredgewidth=2, capsize=5)

        plt.legend(loc='best')

        if save:
            plt.savefig("{0}-wcci.pdf".format(datasets[i]), format="pdf")

        if plot:
            plt.show()
        else:
            plt.close()

if not os.path.isdir(image_path): os.mkdir(image_path)

parser = argparse.ArgumentParser()
parser.add_argument('-i', help='Directory Path', required=True)
parser.add_argument('-r', help='Number of Heade Rows', required=False, type=int)
parser.add_argument('-p', help='Plot Graphs', action='store_true', required=False)
parser.add_argument('-s', help='Save Graphs', action='store_true', required=False)
args = parser.parse_args()

folder = args.i
rows = args.r
plot_flag = args.p
save_flag = args.s

analyse(folder, rows, plot_flag, save_flag)