import argparse
import numpy as np
from os import listdir
from os.path import isfile, join
import utils

def analyse (folder, rows):
    if folder.endswith("/"):
        folder = folder[:-1]

    headerRows = rows
    if rows == None:
        headerRows = 4

    files = [f for f in listdir(folder) if isfile(join(folder, f)) and not f.startswith('.') and f.endswith(".csv") and not f.startswith('analysis-')]
    if len(files) > 1:
        files = sorted(files, key=lambda x: int(x[:-4].split("-l")[-1]))
    else:
        files = sorted(files)

    datasets, folds, headers = utils.read_header(files, folder, headerRows)

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