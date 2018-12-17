import argparse
import pandas as pd
from scipy.io import arff
from os import listdir
from os.path import isfile, join
from sklearn import preprocessing
import os
import utils


def normalize(train_folder, test_folder, output_train_folder, output_test_folder, norm_type):
    train_files = [f for f in listdir(train_folder) if isfile(join(train_folder, f)) and not f.startswith('.')
                   and not f.endswith(".true")]
    train_files = sorted(train_files)

    test_files = [f for f in listdir(test_folder) if isfile(join(test_folder, f)) and not f.startswith('.')
                   and not f.endswith(".true")]
    test_files = sorted(test_files)

    for train, test in zip(train_files, test_files):
        if train.endswith(".arff") or test.endswith(".arff"):
            train_data, train_meta = arff.loadarff(open(join(train_folder, train), 'rb'))
            train_data = pd.DataFrame(train_data)
            train_targets = train_data['class']

            test_data, test_meta = arff.loadarff(open(join(test_folder, test), 'rb'))
            test_data = pd.DataFrame(test_data)
            test_targets = test_data['class']
        else:
            train_data = pd.read_csv(join(train_folder, train), header=None)
            train_targets = train_data.iloc[:, -1].values.astype('int16')

            test_data = pd.read_csv(join(test_folder, test), header=None)
            test_targets = test_data.iloc[:, -1].values.astype('int16')

        if norm_type == 'minmax':
            min_max_scaler = preprocessing.MinMaxScaler().fit(train_data)
            train_data = min_max_scaler.transform(train_data)
            test_data = min_max_scaler.transform(test_data)

        elif norm_type == 'scaler':
            scaler = preprocessing.StandardScaler().fit(train_data)
            train_data = scaler.transform(train_data)
            test_data = scaler.transform(test_data)

        train_data = pd.DataFrame(train_data)
        test_data = pd.DataFrame(test_data)
        train_data[len(train_data.columns) - 1] = train_targets
        test_data[len(test_data.columns) - 1] = test_targets

        if not os.path.isdir(output_train_folder):
            os.mkdir(output_train_folder)

        if not os.path.isdir(output_test_folder):
            os.mkdir(output_test_folder)

        if train.endswith(".arff") or test.endswith(".arff"):
            utils.write_arff_file(train, train_data, train_meta, output_train_folder, train_targets)
            utils.write_arff_file(test, test_data, test_meta, output_test_folder, test_targets)
        else:
            train_data.to_csv(join(output_train_folder, train), sep=',', index=False, header=False)
            test_data.to_csv(join(output_test_folder, train), sep=',', index=False, header=False)


parser = argparse.ArgumentParser()
parser.add_argument('--train', help='Train Directory', required=True)
parser.add_argument('--test', help='Test Directory', required=True)
parser.add_argument('--out-train', help='Output Train Directory', required=True)
parser.add_argument('--out-test', help='Output Test Directory', required=True)
parser.add_argument('--norm', help='Normalization Type', required=False, default='scaler')
args = parser.parse_args()

train_folder = args.train
test_folder = args.test
output_train_folder = args.out_train
output_test_folder = args.out_test
norm_type = args.norm

normalize(train_folder=train_folder, test_folder=test_folder,
          output_train_folder=output_train_folder, output_test_folder=output_test_folder,
          norm_type=norm_type)
