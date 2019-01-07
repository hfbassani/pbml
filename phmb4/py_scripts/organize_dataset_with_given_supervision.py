import argparse
import pandas as pd
import numpy as np
from scipy.io import arff
import os
from os.path import join
import random
import utils


def balanced_sampling(data, supervision_r):
    unique_labels = data['class'].unique()
    labels = data['class']

    labeled_len = int(len(labels) * supervision_r)
    sample_per_class = int(labeled_len / len(unique_labels))

    labeled_samples_indices = []

    for target in unique_labels:
        samples_target = data.index[data["class"] == target].tolist()
        labeled_samples_indices += random.sample(samples_target, int(sample_per_class))

    return labeled_samples_indices


def create_arff(arff_file_name, folder_path, output_path, supervision_r, balanced):
    data, meta = arff.loadarff(open(join(folder_path, arff_file_name), 'rb'))
    data = pd.DataFrame(data)

    saved_labels = data['class'].unique()
    labels = data['class'].values.astype(np.int)

    curr_labels = do_sample(balanced, data, labels, supervision_r)

    data['class'] = curr_labels.astype(np.str)
    curr_labels = set(curr_labels)
    utils.write_arff_file(arff_file_name, data, meta, output_path, curr_labels)
    data = data[data["class"] != str(999)]
    utils.write_arff_file("sup_" + arff_file_name, data, meta, output_path, saved_labels)


def create_default(file_name, folder_path, output_path, supervision_r, balanced):
    data = pd.read_csv(join(folder_path, file_name), sep=",", header=None)
    data = pd.DataFrame(data)

    data = data.rename(columns={(len(data.columns) - 1): "class"})
    labels = data['class']

    curr_labels = do_sample(balanced, data, labels, supervision_r)

    data['class'] = curr_labels.astype(np.int)

    data.to_csv(join(output_path, file_name), sep=',', index=False, header=False)

    data = data[data['class'] != 999]
    data.to_csv(join(output_path, "sup_" + file_name), sep=',', index=False, header=False)


def do_sample(balanced, data, labels, supervision_r):
    if balanced:
        labeled_samples_indices = balanced_sampling(data, supervision_r)
        curr_labels = np.copy(labels)
        curr_labels[~labels.index.isin(labeled_samples_indices)] = 999

    else:
        while True:
            rng = np.random.RandomState(random.randint(1, 20000))
            random_unlabeled_points = rng.rand(len(labels)) >= supervision_r
            curr_labels = np.copy(labels)
            curr_labels[random_unlabeled_points] = 999

            if len(labels) - len(curr_labels[random_unlabeled_points]) >= len(labels) * supervision_r:
                break

    return curr_labels


parser = argparse.ArgumentParser()
parser.add_argument('-i', help='Input Directory', required=True)
parser.add_argument('-s', help='Percentage of Supervision', nargs='+', required=True, type=float)
parser.add_argument('-b', help='Balanced Sampling', action='store_true', required=False)
parser.add_argument('-n', help='Number of Datasets Sampled', required=False, type=int, default=1)
args = parser.parse_args()

folder_path = args.i
folder_path = utils.check_directory(folder_path)
balanced = args.b

for supervision in args.s:
    for dataset_n_number in xrange(args.n):
        output_path = folder_path

        if output_path.endswith("/"):
            output_path = output_path[:-1]

        if args.n > 1:
            output_path += "-n" + dataset_n_number + "-"

        output_path += "S" + ('%.2f' % supervision).split(".")[1]

        if not os.path.isdir(output_path):
            os.mkdir(output_path)

        files = [f for f in os.listdir(folder_path) if
                 os.path.isfile(join(folder_path, f)) and not f.startswith('.') and not f.endswith(".true")]
        files = sorted(files)

        for file in files:
            if file.endswith(".arff"):
                create_arff(arff_file_name=file, folder_path=folder_path, output_path=output_path,
                            supervision_r=supervision, balanced=balanced)
            else:
                create_default(file_name=file, folder_path=folder_path, output_path=output_path,
                               supervision_r=supervision, balanced=balanced)

