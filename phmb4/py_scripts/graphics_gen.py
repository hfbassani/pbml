import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import os
import argparse
from sklearn import linear_model

image_path = "plots/"

#---------------------- Synthetic Data Graphs ----------------------#

def plot_noise_graph(ce, save=False, plot=True):
    noise_values = np.linspace(10, 70, num=4)
    noise_labels = ['10%', '30%', '50%', '70%']

    fig, ax = plt.subplots()
    ax.yaxis.grid()
    ax.set_xticklabels(noise_labels)

    title = 'CE x Noise Percentage'
    plt.title(title, fontsize=14)
    plt.plot(noise_values, ce, '-^', color='k', clip_on=False)
    plt.yticks(np.linspace(0, 1, num=11))
    plt.xticks(noise_values)

    check_plot_save(path="{0}{1}.png".format(image_path, title), save=save, plot=plot)

def subplot_noise_graph(ce, ax):
    noise_values = np.linspace(10, 70, num=4)
    noise_labels = ['10%', '30%', '50%', '70%']

    ax.yaxis.grid()
    ax.set_xticklabels(noise_labels)

    title = 'CE x Noise Percentage'
    ax.set_title(title, fontsize=12)
    ax.plot(noise_values, ce, '-^', color='k', clip_on=False)
    ax.set_yticks(np.linspace(0, 1, num=11))
    ax.set_xticks(noise_values)

def plot_samples_graph(ce, save=False, plot=True):
    samples_size = np.linspace(1500, 5500, num=5)

    fig, ax = plt.subplots()
    ax.yaxis.grid()
    ax.set_xlim(1500, 5500)

    title = 'CE x Dataset Size'
    plt.title(title, fontsize=14)
    plt.plot(samples_size, ce, '-^', color='k', clip_on=False)
    plt.yticks(np.linspace(0, 1, num=11))
    plt.xticks(samples_size)

    check_plot_save(path="{0}{1}.png".format(image_path, title), save=save, plot=plot)

def subplot_samples_graph(ce, ax):
    samples_size = np.linspace(1500, 5500, num=5)

    ax.yaxis.grid()
    ax.set_xlim(1500, 5500)

    title = 'CE x Dataset Size'
    ax.set_title(title, fontsize=12)
    ax.plot(samples_size, ce, '-^', color='k', clip_on=False)
    ax.set_yticks(np.linspace(0, 1, num=11))
    ax.set_xticks(samples_size)

def plot_dimensions_graph(ce, save=False, plot=True):
    x = [5, 10, 15, 20, 25, 50, 75]

    fig, ax = plt.subplots()
    ax.yaxis.grid()
    ax.set_xlim(5, 75)

    title = 'CE x Number of Dimensions'
    plt.title(title, fontsize=14)
    plt.plot(x, ce, "-^", color='k', clip_on=False)
    plt.xticks(np.linspace(5, 75, num=8))
    plt.yticks(np.linspace(0, 1, num=11))

    check_plot_save(path="{0}{1}.png".format(image_path, title), save=save, plot=plot)

def subplot_dimensions_graph(ce, ax):
    x = [5, 10, 15, 20, 25, 50, 75]

    ax.yaxis.grid()
    ax.set_xlim(5, 75)

    title = 'CE x Number of Dimensions'
    ax.set_title(title, fontsize=12)
    ax.plot(x, ce, "-^", color='k', clip_on=False)
    ax.set_xticks(np.linspace(5, 75, num=8))
    ax.set_yticks(np.linspace(0, 1, num=11))

def plot_irrelevant_dims_graph(ce, save=False, plot=True):
    irrelevant_dims_size = np.linspace(0, 5, num=6)

    fig, ax = plt.subplots()
    ax.yaxis.grid()
    ax.set_xlim(0, 5)

    title = 'CE x Number of Irrelevant Dimensions'
    plt.title(title, fontsize=14)
    plt.plot(irrelevant_dims_size, ce, "-^", color='k', clip_on=False)
    plt.xticks(irrelevant_dims_size)
    plt.yticks(np.linspace(0, 1, num=11))

    check_plot_save(path="{0}{1}.png".format(image_path, title), save=save, plot=plot)

def subplot_irrelevant_dims_graph(ce, ax):
    irrelevant_dims_size = np.linspace(0, 5, num=6)

    ax.yaxis.grid()
    ax.set_xlim(0, 5)

    title = 'CE x Number of Irrelevant Dimensions'
    ax.set_title(title, fontsize=12)
    ax.plot(irrelevant_dims_size, ce, "-^", color='k', clip_on=False)
    ax.set_xticks(irrelevant_dims_size)
    ax.set_yticks(np.linspace(0, 1, num=11))

def plot_synthetic_data_graphs(fileName, save=False, plot=True):
    results = get_headers(fileName)

    plot_dimensions_graph(results["max_value"][:7], save=save, plot=plot)
    plot_noise_graph(results["max_value"][7:11], save=save, plot=plot)
    plot_samples_graph(results["max_value"][11:16], save=save, plot=plot)
    plot_irrelevant_dims_graph(results["max_value"][16:], save=save, plot=plot)

def subplot_synthetic_data_graphs(fileName, save=False, plot=True):
    results = get_headers(fileName)

    fig, axs = plt.subplots(nrows=2, ncols=2)

    subplot_dimensions_graph(results["max_value"][:7], axs[0, 0])
    subplot_noise_graph(results["max_value"][7:11], axs[0, 1])
    subplot_samples_graph(results["max_value"][11:16], axs[1, 0])
    subplot_irrelevant_dims_graph(results["max_value"][16:], axs[1, 1])

    fig.tight_layout()

    check_plot_save(path="{0}best_values.png".format(image_path), save=save, plot=plot)

#-------------------------------------------------------------------#
#---------------------- Common Methods -----------------------------#

def check_plot_save (path, save, plot):
    if save:
        plt.savefig(path)

    if plot:
        plt.show()
    else:
        plt.close()

def plot_fit_linear(to_plot, x, y):
    x = x.values
    y = y.values.astype(np.float)

    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(x.reshape(-1, 1), y.reshape(-1, 1))

    # Make predictions using the testing set
    fit = regr.predict(x.reshape(-1, 1))

    to_plot.plot(x, fit, color='r', clip_on=False, linewidth=6)

def plot_x_y(x, y, title, marker="o", color='b', fontSize=12, save=False, plot=True):
    fig, ax = plt.subplots()
    ax.yaxis.grid()
    ax.set_ylim([0, 1])

    plt.title(title, fontsize=fontSize)
    plt.plot(x, y, marker, color=color, clip_on=False)
    plt.yticks(np.linspace(0, 1, num=11))

    plot_fit_linear(plt, x, y)

    check_plot_save(path="{0}{1}.png".format(image_path, title), save=save, plot=plot)

def subplot_x_y(ax, x, y, title, marker="o", color='b', fontSize=12):
    ax.yaxis.grid()
    ax.set_ylim([0, 1])

    ax.set_title(title, fontsize=fontSize)
    ax.plot(x, y, marker, color=color, clip_on=False)
    ax.set_yticks(np.linspace(0, 1, num=11))

    plot_fit_linear(ax, x, y)

def get_headers(fileName):
    headers = pd.read_csv(fileName, nrows=9, header=None)
    headers = headers.transpose()
    headers = headers.rename(columns=headers.iloc[0])
    headers = headers.drop([0])
    headers = headers.dropna(axis=0, how='any')
    headers = headers.astype(np.float64)

    return headers

def get_params_and_results(fileName):
    results = pd.read_csv(fileName, skiprows=10, header=None)

    firstParamIndex = results.iloc[0]
    firstParamIndex = firstParamIndex[firstParamIndex == "a_t"].index[0]
    params = results.drop(results.columns[range(firstParamIndex)], axis=1)
    params = params.rename(columns=params.iloc[0])
    params = params.drop([0])
    params = params.astype(np.float64)

    results = results.drop(results.columns[range(firstParamIndex, len(results.columns))], axis=1)
    results = results.drop(results.columns[0], axis=1)
    results = results.rename(columns=results.iloc[0])
    results = results.drop([0])

    return params, results

#-------------------------------------------------------------------#
#---------------------- Parameters Graphs --------------------------#

def plot_params_results(fileName, paramsToPlot=None, save=False, plot=True):

    params, results = get_params_and_results(fileName)

    if paramsToPlot == None:
        paramsToPlot = params.columns

    for param in paramsToPlot:
        for result in results.columns:
#            if "Accuracy" in result:
            plot_x_y(params[param], results[result], "{0} - {1}".format(param, result), save=save, plot=plot)


def subplot_params_results(fileName, paramsToPlot=None, save=False, plot=True):
    params, results = get_params_and_results(fileName)

    if paramsToPlot == None:
        paramsToPlot = params.columns

    if len(paramsToPlot) == 1:
        plot_params_results(fileName, paramsToPlot, save=save, plot=plot)

    ncols = 2
    nrows = int(np.ceil(len(paramsToPlot) / 2.0))
    rest = len(paramsToPlot) % 2 == 1

    for result in results.columns:
        row = 0
        col = 0

        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7,7))
        fig.suptitle(result, fontsize=14, y=0.99)

        for param in paramsToPlot:
            if row == nrows:
                break

            if len(axs.shape) == 1:
                fig.set_size_inches((8,4))
                subplot_x_y(axs[col], params[param], results[result], param)
            else:
                subplot_x_y(axs[row, col], params[param], results[result], param)

            if col < ncols - 1:
                col += 1
            else:
                col = 0
                row += 1

        if rest:
            fig.delaxes(axs[row, col])

        fig.tight_layout()

        check_plot_save(path="{0}{1}.png".format(image_path, result), save=save, plot=plot)

def plot_gammas_vs_hthresholds(filename, save=False, plot=True):
    headers = get_headers(fileName)
    params, _ = get_params_and_results(fileName)

    indexes = list(headers["index_set"])
    indexes = map(int, indexes)
    indexes = set(indexes)

    gammas = []
    if "gamma" in paramsToPlot:
        values = list(params["gamma"])
        for index in indexes:
            gammas.append(values[index])

    h_threshs = []
    if "h_threshold" in paramsToPlot:
        values = list(params["h_threshold"])
        for index in indexes:
            h_threshs.append(values[index])

    if len(gammas) > 0 and len(h_threshs) > 0:
        h_order = np.argsort(gammas)
        gammas = np.sort(gammas)
        for i in xrange(len(gammas)):
            h = []
            for j in xrange(len(gammas)):
                h.append(np.exp(- (j / gammas[i])))

            fig, ax = plt.subplots()
            ax.yaxis.grid()
            ax.set_ylim([0, 1])
            ax.set_xlim([1, len(gammas)])

            h_thresh_x = [1, len(gammas)]
            h_thresh_y = [h_threshs[h_order[i]]] * 2
            plt.plot(h_thresh_x, h_thresh_y, "-", color='r', clip_on=False)

            plt.title("Gamma {0} | H_Thresh {1}".format(gammas[i], h_threshs[h_order[i]]), fontsize=14)
            plt.plot(np.linspace(1, len(gammas), num=len(gammas)), h, "-o", color='b', clip_on=False)
            plt.xticks(np.arange(1, len(gammas) + 1))

            plt.yticks(np.linspace(0, 1, num=11))

            check_plot_save(path="{0}gammas_x_hthresh{1}.png".format(image_path, i), save=save, plot=plot)

#-------------------------------------------------------------------#
if not os.path.isdir(image_path): os.mkdir(image_path)

parser = argparse.ArgumentParser()
parser.add_argument('-i', help='File Path', required=True)
parser.add_argument('-r', help='Number of Heade Rows', required=False, type=int)
parser.add_argument('-p', help='Parameters to plot', nargs='*', required=False, type=str, default=None)
parser.add_argument('-s', help='Plot synthethic data', action='store_true', required=False)
parser.add_argument('--plot', help='Normal plot', action='store_true', required=False)
parser.add_argument('--subplot', help='Subplot', action='store_true', required=False)
parser.add_argument('--save', help='Subplot', action='store_true', required=False)
parser.add_argument('--show', help='Subplot', action='store_true', required=False)

args = parser.parse_args()

fileName = args.i
header_rows = args.r
paramsToPlot = args.p # ["a_t", "lp", "dsbeta", "age_wins", "e_b", "e_n", "epsilon_ds", "minwd","epochs", "push_rate", "supervision_rate"]

synthetic_plot = args.s
plot = args.plot
subplot = args.subplot

save = args.save
show = args.show

if synthetic_plot:
    if plot:
        plot_synthetic_data_graphs(fileName=fileName, save=save, plot=show)
    elif subplot:
        subplot_synthetic_data_graphs(fileName=fileName, save=save, plot=show)

else:
    if plot:
        plot_params_results(fileName=fileName, paramsToPlot=paramsToPlot, save=save, plot=show)
    elif subplot:
        subplot_params_results(fileName=fileName, paramsToPlot=paramsToPlot, save=save, plot=show)

