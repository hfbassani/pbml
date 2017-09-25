import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_noise_graph (ce, savePlots=False):
    noise_values = np.linspace(10, 70, num=4)
    noise_labels = ['10%', '30%', '50%', '70%']

    fig, ax = plt.subplots()
    ax.yaxis.grid()
    ax.set_xticklabels(noise_labels)

    title = 'CE x Noise Percentage'
    plt.title(title, fontsize=14)
    plt.plot(noise_values, ce, '-^', color='k', clip_on=False)
    plt.yticks(np.linspace(0, 1, num=11))
    
    if savePlots:
        plt.savefig("{0}.png".format(title))
        
    plt.show()

def plot_samples_graph (ce, savePlots=False):
    samples_size = np.linspace(1500, 5500, num=5)

    plot_x_y(samples_size, ce, 'CE x Dataset Size', '-^', 'k', 14, savePlots=savePlots)
    
def plot_dimensions_graph (ce, savePlots=False):
    x = [5, 10, 15, 20, 25, 50, 75]
    
    fig, ax = plt.subplots()
    ax.yaxis.grid()
    ax.set_xlim(5, 75)

    title = 'CE x Number of Dimensions'
    plt.title(title, fontsize=14)
    plt.plot(x, ce, "-^", color='k', clip_on=False)
    plt.xticks(np.linspace(5, 75, num=8))
    plt.yticks(np.linspace(0, 1, num=11))
    
    if savePlots:
        plt.savefig("{0}.png".format(title))
        
    plt.show()
    
def plot_irrelevant_dims_graph (ce, savePlots=False):
    irrelevant_dims_size = np.linspace(0, 5, num=6)

    plot_x_y(irrelevant_dims_size, ce, 'CE x Number of Irrelevant Dimensions', '-^', 'k', 14, savePlots=savePlots)

def plot_x_y(x, y, title, marker="o", color='b', fontSize=12, savePlots=False):
    fig, ax = plt.subplots()
    ax.yaxis.grid()
    ax.set_ylim([0, 1])

    plt.title(title, fontsize=fontSize)
    plt.plot(x, y, marker, color=color, clip_on=False)
    plt.yticks(np.linspace(0, 1, num=11))
    
    if savePlots:
        plt.savefig("{0}.png".format(title))
        
    plt.show()
    

def get_headers(fileName):
    headers = pd.read_csv(fileName, nrows=4, header=None)
    headers = headers.transpose()
    headers = headers.rename(columns=headers.iloc[0])
    headers = headers.drop([0])
    headers = headers.dropna(axis=0, how='any')
    headers = headers.astype(np.float64)

    return headers

def plot_synthetic_data_graphs(fileName, savePlots=False):
    results = get_headers(fileName)
    
    plot_dimensions_graph(results["max_value"][:7], savePlots=savePlots)
    plot_noise_graph(results["max_value"][7:11], savePlots=savePlots)
    plot_samples_graph(results["max_value"][11:16], savePlots=savePlots)
    plot_irrelevant_dims_graph(results["max_value"][16:], savePlots=savePlots)
    
def plot_params_results(fileName, paramsToPlot = None, savePlots = False):

    results = pd.read_csv(fileName, skiprows=5, header=None)

    headers = get_headers(fileName)

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
    
    if paramsToPlot == None:
        paramsToPlot = params.columns
        
    indexes = list(headers["index_set"])
    indexes = map(int, indexes)
    indexes = set(indexes)

    gammas = []
    h_threshs = []
    for param in paramsToPlot:
        for result in results.columns:
            x = 1
            plot_x_y(params[param], results[result], "{0} - {1}".format(param, result), savePlots=savePlots)
            
        if param == "gamma":
            values = list(params[param])
            for index in indexes:
                gammas.append(values[index])

        if param == "h_threshold":
            values = list(params[param])
            for index in indexes:
                h_threshs.append(values[index])


    if len(gammas) > 0 and len(h_threshs) > 0:
        h_order = np.argsort(gammas)
        gammas = np.sort(gammas)
        for i in xrange(len(gammas)):
            h = []
            for j in xrange(len(gammas)):
                h.append(np.exp( - (j / gammas[i])))

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
            
            if savePlots:
                plt.savefig("gammas_x_hthresh{0}.png".format(i))
                
            plt.show()
            

fileName = "../outputMetrics/results_ParamsNodeDelNNSim500_0_at_order_seq_finish.csv"

plot_synthetic_data_graphs(fileName=fileName, savePlots=True)

paramsToPlot = ["a_t", "lp", "gamma", "h_threshold"]
plot_params_results(fileName=fileName, paramsToPlot=paramsToPlot, savePlots=True)

