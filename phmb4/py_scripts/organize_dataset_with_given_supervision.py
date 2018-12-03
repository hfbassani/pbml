import argparse
import pandas as pd
import numpy as np
from scipy.io import arff
import os
from os.path import join
import random
import utils


def create_arff(arff_file_name, folder_path, output_path, supervision_r):
    data, meta = arff.loadarff(open(join(folder_path, arff_file_name), 'rb'))
    data = pd.DataFrame(data)

    saved_labels = data['class'].unique()
    labels = data['class']

    while True:
        rng = np.random.RandomState(random.randint(1, 20000))
        random_unlabeled_points = rng.rand(len(labels)) >= supervision_r
        curr_labels = np.copy(labels)
        curr_labels[random_unlabeled_points] = str(999)

        if len(labels) - len(curr_labels[random_unlabeled_points]) >= len(labels) * supervision_r:
            break

    data['class'] = curr_labels
    curr_labels = set(curr_labels)
    utils.write_arff_file(arff_file_name, data, meta, output_path, curr_labels)
    data = data[data["class"] != str(999)]
    utils.write_arff_file("sup_" + arff_file_name, data, meta, output_path, saved_labels)


def create_default(file_name, folder_path, output_path, supervision_r):
    data = pd.read_csv(join(folder_path, file_name), sep=",", header=None)
    data = pd.DataFrame(data, dtype=float)

    labels = data.iloc[:, -1].values.astype(np.int)

    while True:
        rng = np.random.RandomState(random.randint(1, 20000))
        random_unlabeled_points = rng.rand(len(labels)) >= supervision_r
        curr_labels = np.copy(labels)
        curr_labels[random_unlabeled_points] = 999

        if len(labels) - len(curr_labels[random_unlabeled_points]) >= len(labels) * supervision_r:
            break

    data[len(data.columns) - 1] = curr_labels

    data.to_csv(join(output_path, file_name), sep=',', index=False, header=False)

    data = data[data[len(data.columns) - 1] != 999]
    data.to_csv(join(output_path, "sup_" + file_name), sep=',', index=False, header=False)


parser = argparse.ArgumentParser()
parser.add_argument('-i', help='Input Directory', required=True)
parser.add_argument('-s', help='Percentage of Supervision', nargs='+', required=True, type=float)
args = parser.parse_args()

folder_path = args.i
folder_path = utils.check_directory(folder_path)

for supervision in args.s:
    output_path = folder_path

    if output_path.endswith("/"):
        output_path = output_path[:-1]

    output_path += "S" + ('%.2f' % supervision).split(".")[1]

    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    files = [f for f in os.listdir(folder_path) if
             os.path.isfile(join(folder_path, f)) and not f.startswith('.') and not f.endswith(".true")]
    files = sorted(files)

    for file in files:
        if file.endswith(".arff"):
            create_arff(arff_file_name=file, folder_path=folder_path,
                        output_path=output_path, supervision_r=supervision)
        else:
            create_default(file_name=file, folder_path=folder_path,
                           output_path=output_path, supervision_r=supervision)

