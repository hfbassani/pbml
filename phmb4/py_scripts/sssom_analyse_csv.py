import argparse
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from os.path import isfile, join
import itertools
import utils

def analyse (folder, rows, plot, save, extension, extra_results, crop):

    if folder.endswith("/"):
        folder = folder[:-1]

    datasets, method, line, plot_means, plot_stds = summarize(folder, rows)

    if save or plot:
        plot_graph(plot_means, plot_stds, datasets, plot, save, extension, folder, extra_results, crop)

    outputFile = open(join(folder, "analysis-" + folder + ".csv"), "w+")
    outputFile.write(line)

def summarize(folder, rows):
    headerRows = rows
    if rows == None:
        headerRows = 9

    files = [f for f in listdir(folder) if isfile(join(folder, f)) and not f.startswith('.') and f.endswith(".csv")
             and not f.startswith('analysis-') and not f.startswith('parameters-')]

    if len(files) > 1:
        files = sorted(files, key=lambda x: int(x[:-4].split("-l")[-1]))
    else:
        files = sorted(files)

    method = files[0].split("-l")[0]

    datasets, folds, headers = utils.read_header(files, folder, headerRows)

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

    return datasets, method, line, np.array(plot_means), np.array(plot_stds)


def plot_graph(means, stds, datasets, plot, save, extensions, folder, extra_results, crop):
    percentage_values = np.linspace(1, 100, num=7)
    percentage_labels = ['1%', '5%', '10%', '25%', '50%', '75%', '100%']

    plot_means = np.transpose(means)
    plot_stds = np.transpose(stds)

    markers = itertools.cycle(('x', 'D', '*'))
    linestyles = itertools.cycle(('-.', '--', ':'))

    for i in xrange(len(datasets)):
        fig, ax = plt.subplots()
        ax.yaxis.grid()
        ax.set_ylim([0, 1])
        ax.set_xticklabels(percentage_labels[:plot_means.shape[1]])

        title = datasets[i].split("_")[1].capitalize()

        plt.rc('font', family='serif')
        plt.title(title, fontsize=14)
        plt.yticks(np.linspace(0, 1, num=11))
        plt.xticks(percentage_values)

        for extra in extra_results:
            datasets_extra, method_extra, line_extra, plot_means_extra, plot_stds_extra = summarize(extra, 4)

            plt.errorbar(percentage_values, plot_means_extra[:, i], plot_stds_extra[:, i], label=method_extra,
                         linestyle=linestyles.next(), marker=markers.next(), clip_on=False, markeredgewidth=2, capsize=5)

        plt.errorbar(percentage_values, plot_means[i], plot_stds[i], label='SS-SOM',
                     linestyle='-', marker='o', clip_on=False, markeredgewidth=2, capsize=5)

        plt.legend(loc='best')

        if save:
            for extension in extensions:
                plotPath = join(folder,"{0}-wcci.{1}".format(title, extension))

                if crop:
                    plt.savefig(plotPath, bbox_inches='tight', pad_inches = 0)
                else:
                    plt.savefig(plotPath)

        if plot:
            plt.show()
        else:
            plt.close()

parser = argparse.ArgumentParser()
parser.add_argument('-i', help='Directory Path', required=True)
parser.add_argument('-r', help='Number of Heade Rows', required=False, type=int)
parser.add_argument('-p', help='Plot Graphs', action='store_true', required=False)
parser.add_argument('-s', help='Save Graphs', action='store_true', required=False)
parser.add_argument('-e', help='Extension', nargs='+', required=False, type=str, default=[])
parser.add_argument('-a', help='Additional .csv files to plot', nargs='+', required=False, type=str, default=[])
parser.add_argument('-c', help='Crop PDF', action='store_true', required=False)

args = parser.parse_args()

if args.s and not args.e:
    parser.error("[-s save] requires [-e extension(s)].")

if args.c and (not args.s or not args.e):
    parser.error("[-c crop] requires [-s save] and [-e extension(s)].")

folder = args.i
rows = args.r
plot_flag = args.p
save_flag = args.s
extensions = args.e
extra_results = sorted(args.a)
crop = args.c

analyse(folder, rows, plot_flag, save_flag, extensions, extra_results, crop)