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
        random_unlabeled_points = rng.rand(len(labels)) > supervision_r
        curr_labels = np.copy(labels)
        curr_labels[random_unlabeled_points] = str(999)

        if len(labels) - len(curr_labels[random_unlabeled_points]) > len(labels) * supervision_r:
            break
    data['class'] = curr_labels
    curr_labels = set(curr_labels)
    write_file(arff_file_name, data, meta, output_path, curr_labels)
    data = data[data["class"] != str(999)]
    write_file("sup_" + arff_file_name, data, meta, output_path, saved_labels)


def write_file(arff_file_name, data, meta, output_path, saved_labels):
    new_file = open(join(output_path, arff_file_name), 'w+')
    new_file.write("@relation {0}\n".format(meta.name))
    for i in xrange(len(meta.names())):
        attr = meta.names()[i]
        if attr != "class":
            new_file.write("@attribute {0} {1}\n".format(attr, utils.get_type(meta.types()[i])))
        else:
            new_file.write("@attribute {0} {{".format(attr))
            new_file.write("{0}".format(",".join(saved_labels)))
            new_file.write("}\n")
    new_file.write("@data\n")
    for _, row in data.iterrows():
        new_file.write(",".join(map(str, row)) + "\n")


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

    for file in sorted(os.listdir(folder_path)):
        if file.endswith(".arff"):
            create_arff(arff_file_name=file, folder_path=folder_path,
                        output_path=output_path, supervision_r=supervision)
        else:
            create_default(file_name=file, folder_path=folder_path,
                           output_path=output_path, supervision_r=supervision)

