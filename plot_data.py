from imports import *
from classes.Plotter import Plotter

titles = ["Active Learning Query Functions vs. Random Sampling",
          "Variation Ratio across different Query-Pool Sizes",
          "Heuristics compared to Variation Ratio"]
ylabels = ['Accuracy (%)', 'ROCAUC score']
lines = [
    [
        mlines.Line2D([], [], color='red', marker='s', linestyle='None',
                      markersize=10, label='Random Sampling'),
        mlines.Line2D([], [], color='green', marker='o', linestyle='None',
                      markersize=10, label='Variation Ratio'),
        mlines.Line2D([], [], color='blue', marker='v', linestyle='None',
                      markersize=10, label='Predictive Entropy'),
        mlines.Line2D([], [], color='black', marker='x', linestyle='None',
                      markersize=10, label='BALD'),
    ],
    [
        mlines.Line2D([], [], color='red', marker='s', linestyle='None',
                      markersize=10, label='Query-pool size of 0.5\%'),
        mlines.Line2D([], [], color='green', marker='o', linestyle='None',
                      markersize=10, label='Query-pool size of 1\%'),
        mlines.Line2D([], [], color='blue', marker='v', linestyle='None',
                      markersize=10, label='Query-pool size of 5\%'),
    ],
    [
        mlines.Line2D([], [], color='red', marker='s', linestyle='None',
                      markersize=10, label='Variation Ratio'),
        mlines.Line2D([], [], color='green', marker='o', linestyle='None',
                      markersize=10, label='Variation Ratio + RET'),
        mlines.Line2D([], [], color='blue', marker='s', linestyle='None',
                      markersize=10, label='Variation Ratio + RECT'),
        mlines.Line2D([], [], color='black', marker='o', linestyle='None',
                      markersize=10, label='Variation Ratio + SUD'),
    ]
]


def main():
    """Initialise a plotter for the input experiment and make it plot the data"""
    if sys.argv[1] == "1":
        plotter_accuracy = Plotter(3, 'random_vs_al_accuracy.txt', lines[0])
        plotter_roc = Plotter(3, 'random_vs_al_roc.txt', lines[0])
        plotter_accuracy.plot_data(
            'accuracy', 593, 85, ylabels[0], 'SST: ' + titles[0], 'random_vs_al', [5, 5, 5, 5])
        plotter_roc.plot_data(
            'rocauc', 593, 85, ylabels[1], 'SST: ' + titles[0], 'random_vs_al', [5, 5, 5, 5])
    elif sys.argv[1] == "2":
        plotter_accuracy = Plotter(3, 'scaling_accuracy.txt', lines[1])
        plotter_roc = Plotter(3, 'scaling_roc.txt', lines[1])
        plotter_accuracy.plot_data(
            'accuracy', 593, 85, ylabels[0], 'SST: ' + titles[1], 'scaling', [10, 5, 1])
        plotter_roc.plot_data(
            'rocauc', 593, 85, ylabels[1], 'SST: ' + titles[1], 'scaling', [10, 5, 1])
    elif sys.argv[1] == "3":
        plotter_accuracy = Plotter(3, 'heuristics_accuracy.txt', lines[2])
        plotter_roc = Plotter(3, 'heuristics_roc.txt', lines[2])
        plotter_accuracy.plot_data(
            'accuracy', 593, 85, ylabels[0], 'SST: ' + titles[2], 'heuristics', [5, 5, 5, 5])
        plotter_roc.plot_data(
            'rocauc', 593, 85, ylabels[1], 'SST: ' + titles[2], 'heuristics', [5, 5, 5, 5])


if __name__ == "__main__":
    main()
