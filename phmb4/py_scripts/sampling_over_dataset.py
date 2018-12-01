import argparse
import pandas as pd
import numpy as np
from scipy.io import arff
import os
from os.path import join
import random
import utils


def read_arff(arff_file_name, folder_path, output_path, sampling_rate):
    data, meta = arff.loadarff(open(join(folder_path, arff_file_name), 'rb'))
    data = pd.DataFrame(data)


    new_data = sample(data, sampling_rate)

    labels = data['class'].unique()

    utils.write_arff_file(arff_file_name, new_data, meta, output_path, labels)


def read_default(file_name, folder_path, output_path, sampling_rate):
    data = pd.read_csv(join(folder_path, file_name), sep=",", header=None)
    data = pd.DataFrame(data, dtype=float)

    new_data = sample(data, sampling_rate)

    new_data.to_csv(join(output_path, file_name), sep=',', index=False, header=False)


def sample(data, sampling_rate):
    while True:
        rng = np.random.RandomState(random.randint(1, 20000))
        new_samples = rng.rand(len(data)) <= sampling_rate
        new_data = data[new_samples]

        if len(new_data) <= len(data) * sampling_rate:
            return new_data


parser = argparse.ArgumentParser()
parser.add_argument('-i', help='Input Directory', required=True)
parser.add_argument('-p', help='Percentage of Supervision', nargs='+', required=True, type=float)
args = parser.parse_args()

folder_path = args.i
folder_path = utils.check_directory(folder_path)

for sampling_rate in args.p:
    output_path = folder_path

    if output_path.endswith("/"):
        output_path = output_path[:-1]

    output_path += "SP" + ('%.2f' % sampling_rate).split(".")[1]

    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    files = [f for f in os.listdir(folder_path) if
             os.path.isfile(join(folder_path, f)) and not f.startswith('.') and not f.endswith(".true")]
    files = sorted(files)

    for file in files:
        if file.endswith(".arff"):
            read_arff(arff_file_name=file, folder_path=folder_path,
                      output_path=output_path, sampling_rate=sampling_rate)
        else:
            read_default(file_name=file, folder_path=folder_path,
                         output_path=output_path, sampling_rate=sampling_rate)