import argparse
from os import listdir
from os.path import isfile, join, isdir
from scipy.stats import wilcoxon
import utils


def run_test(file_x, file_y, rows_x, rows_y, confidence, test):
    if test == 'wilcoxon':
        _, result_x = utils.get_params_and_results(file_x, rows_x)
        _, result_y = utils.get_params_and_results(file_y, rows_y)

        for column in result_x.columns:
            x = result_x[column].values.astype(float)
            y = result_y[column].values.astype(float)

            statistic, pvalue = wilcoxon(x, y)

            if pvalue > (1 - confidence):
                print ("Statistical Differente Between " + file_x + " and " + file_y + " in " + column)


parser = argparse.ArgumentParser()
parser.add_argument('-d', help='Directories Path', required=False)
parser.add_argument('-f', help='Files Path', nargs='+', required=False)
parser.add_argument('-c', help='Something to Compare', nargs='+', required=True)
parser.add_argument('-t', help='Statistical Test to be applied', required=False, type=str, default="wilcoxon")
parser.add_argument('-r', help='Number of Header Rows', required=False, type=int, default=9)
parser.add_argument('-R', help='Number of Header Rows of comparing .csvs', required=False, type=int, default=9)
parser.add_argument('-i', help='Confidence Interval', required=False, type=int, default=0.95)

args = parser.parse_args()

if not args.d and not args.f:
    parser.error("[-d directories] or [-f files] required.")

test = args.t

if isdir(args.d):
    input_files = [f for f in listdir(args.d) if isfile(join(args.d, f)) and not f.startswith('.') and f.endswith(".csv")
                   and not f.startswith('analysis-') and not f.startswith('parameters-')]
    input_files = sorted(input_files)

    for compare in args.c:
        comparison_files = [f for f in listdir(compare) if isfile(join(compare, f)) and not f.startswith('.') and f.endswith(".csv")
                             and not f.startswith('analysis-') and not f.startswith('parameters-')]

        if len(comparison_files) > 1:
            comparison_files = sorted(comparison_files, key=lambda x: int(x.split(".")[0].split("-l")[-1]))
        else:
            comparison_files = sorted(comparison_files)

        for orig, comp in zip(input_files, comparison_files):
            run_test(join(args.d, orig), join(compare, comp), args.r, args.R, args.i, test)

elif isfile(args.f):
    for compare in args.c:
        run_test(args.f, compare, args.r, args.R, args.i, test)
