from matplotlib.pyplot import ylabel
from imports import *
import numpy as np


class Plotter:
    """ 
    Defines a class that is used query data based on computed uncertainties

    Attributes:
    -----------
    N: int
        The amount of runs to average the experiment over
    file_name: str 
        The name of the file containing the data needed for plotting
    lines: list
        List of the different line-types to be plotted

    Methods
    -------
    plot_data(metric, seed_size, Q, ylabel, title, experiment_name, interval):
        Plots the data to a linegraph with errorbars
    file_to_arrays():
        Processes the strings out of a file to transform them to arrays
    compute_sd(experiment):
        Computes the standard deviation of all points based on an experiment of N runs
    """

    def __init__(self, N, file_name, lines) -> None:
        """
        Parameters:
        -----------
        N: int
            The amount of runs to average the experiment over
        file_name: str 
            The name of the file containing the data needed for plotting
        lines: list
            List of the different line-types to be plotted
        """
        self.N = N
        self.file_name = file_name
        self.lines = lines

    def plot_data(self, metric, seed_size, Q, ylabel, title, experiment_name, interval):
        """
        Plots the data to a linegraph with errorbars

        Parameters:
        -----------
        metric: str
            The metric used to evaluate performance
        seed_size: int
            The initial size of the labeled dataset
        Q: int
            The amount of examples queried in an active learning round
        ylabel: str
            The label of the y-axis
        title: str
            The title of the plot
        experiment_name: str
            The title of the experiment which will be used in the name of the save
        interval: int
            The interval by which to plot points out of the data arrays
        """
        data = self.file_to_arrays()
        fig, ax = plt.subplots()
        for i, experiment in enumerate(data):
            if metric == 'accuracy':
                experiment *= 100
            x = np.linspace(
                seed_size, Q*len(data[0][0]), num=len(experiment[0]))
            y = np.divide(np.sum(experiment, axis=0), self.N)
            sd = self.compute_sd(experiment)
            markers, caps, bars = ax.errorbar(x[::interval[i]], y[::interval[i]], sd[::interval[i]],
                                              color=self.lines[i].get_color(), marker=self.lines[i].get_marker())
            [bar.set_alpha(0.5) for bar in bars]
            [cap.set_alpha(0.5) for cap in caps]

        ax.legend(handles=self.lines)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(r'Size of $\mathcal{L}$')
        ax.set_title(title)
        plt.savefig('./Plots/' + experiment_name + str(metric) + str(Q))

    def file_to_arrays(self):
        """
        Processes the strings out of a file to transform them to arrays

        Returns:
        --------
        data: list
            The arrays containing the values of the different runs/experiments
        """
        data = []
        experiment = []
        with open(self.file_name, 'r') as f:
            for i, line in enumerate(f):
                experiment.append(np.fromstring(line, sep=' '))
                if (i + 1) % self.N == 0 and i != 0:
                    data.append(np.array(experiment))
                    experiment = []
        return data

    def compute_sd(self, experiment):
        """Computes and returns the standard deviation of all points based on an experiment of N runs"""
        sds = []
        for i in range(len(experiment[0])):
            mean = sum(experiment[:, i])/len(experiment[:, i])
            sd = np.sqrt((1/self.N)*sum(np.power(experiment[:, i]-mean, 2)))
            sds.append(sd)
        return sds
