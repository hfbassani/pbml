import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_noise_graph (ce):
    noise_values = np.linspace(10, 70, num=4)
    noise_labels = ['10%', '30%', '50%', '70%']

    fig, ax = plt.subplots()
    ax.yaxis.grid()
    ax.set_xticklabels(noise_labels)

    plt.title('CE Measure x Noise Percentage [LARFDSSOM]', fontsize=14)
    plt.plot(noise_values, ce, '-^', color='k', clip_on=False)
    plt.xticks(noise_values)
    plt.yticks(np.linspace(0, 1, num=11))
    plt.show()

def plot_samples_graph (ce):
    samples_size = np.linspace(1500, 5500, num=5)

    fig, ax = plt.subplots()
    ax.yaxis.grid()

    plt.title('CE Measure x Dataset Size [LARFDSSOM]', fontsize=12)
    plt.plot(samples_size, ce, "-^", color='k', clip_on=False)
    plt.xticks(samples_size)
    plt.yticks(np.linspace(0, 1, num=11))
    plt.show()
    
def plot_dimensions_graph (ce):
    x = [5, 10, 15, 20, 25, 50, 75]
    
    fig, ax = plt.subplots()
    ax.yaxis.grid()
    ax.set_xlim(5, 75)

    plt.title('CE Measure x Number of Dimensions [LARFDSSOM]', fontsize=12)
    plt.plot(x, ce, "-^", color='k', clip_on=False)
    plt.xticks(np.linspace(5, 75, num=8))
    plt.yticks(np.linspace(0, 1, num=11))
    plt.show()
    
def plot_irrelevant_dims_graph (ce):
    irrelevant_dims_size = np.linspace(0, 5, num=6)
    
    fig, ax = plt.subplots()
    ax.yaxis.grid()

    plt.title('CE Measure x Number of Number of Irrelevant Dimensions [LARFDSSOM]', fontsize=12)
    plt.plot(irrelevant_dims_size, ce, "-^", color='k', clip_on=False)
    plt.xticks(irrelevant_dims_size)
    plt.yticks(np.linspace(0, 1, num=11))
    plt.show()


def plot_x_y(x, y, title):

    fig, ax = plt.subplots()
    ax.yaxis.grid()

    plt.title(title, fontsize=12)
    plt.plot(x, y, "o", color='b', clip_on=False)
    plt.show()

def plot_synthetic_data_graphs(fileName):
    results = pd.read_csv(fileName, nrows=3, header=None)
    results = results.transpose()
    results = results.rename(columns=results.iloc[0])
    results = results.drop([0])
    results = results.dropna(axis=0, how='any')
    results = results.astype(np.float64)
    
#    print results["max_value"]
#    print results["max_value"][:7]
#    print results["max_value"][7:11]
#    print results["max_value"][11:16]
#    print results["max_value"][16:]
    
    plot_dimensions_graph(results["max_value"][:7])
    plot_noise_graph(results["max_value"][7:11])
    plot_samples_graph(results["max_value"][11:16])
    plot_irrelevant_dims_graph(results["max_value"][16:])
    
def plot_params_results(fileName, paramsToPlot = None):
    results = pd.read_csv(fileName, skiprows=4, header=None)

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
        
    for param in paramsToPlot:
        for result in results.columns:
            plot_x_y(params[param], results[result], "{0} - {1}".format(param, result))
            
def plot_h_graph():
    gamma_values = np.linspace(0.1, 1, num=19)
    
    for gamma in gamma_values:
        h = []
        for i in xrange(19):
            h.append(np.exp( - (i / gamma)))
            
            
        fig, ax = plt.subplots()
        ax.yaxis.grid()

        plt.title("Gamma {0} x H".format(gamma))
        plt.plot(gamma_values, h, "-o", color='b', clip_on=False)
        plt.yticks(np.linspace(0, 1, num=21))
        plt.show()
    
            
fileName = "../outputMetrics/v1_htrhesh0-1.csv"

#plot_synthetic_data_graphs(fileName=fileName)

paramsToPlot = ["gamma", "h_threshold"]
plot_params_results(fileName=fileName, paramsToPlot=paramsToPlot)

#plot_h_graph()

