from imports import *
import numpy as np


def file_to_arrays(file_name, n):
    arrays = []
    experiment = []
    with open(file_name, 'r') as f:
        for i, line in enumerate(f):
            experiment.append(np.fromstring(line, sep=' '))
            if (i + 1) % n == 0 and i != 0:
                arrays.append(np.array(experiment))
                experiment = []
    return arrays


def compute_deficiency(ref, al):
    nom = 0
    denom = 0
    for i, x in enumerate(al):
        nom += (max(ref) - x)
        denom += (max(ref) - ref[i])
    return nom/denom


def main():
    arrays = []
    if sys.argv[1] == "1":
        print("ye")
        arrays = file_to_arrays('random_vs_al_accuracy.txt', 3)
    elif sys.argv[1] == "2":
        arrays = file_to_arrays('scaling_accuracy.txt', 3)
    elif sys.argv[1] == "3":
        arrays = file_to_arrays('heuristics_accuracy', 3)
    averaged_arrays = []
    for arr in arrays:
        averaged_arrays.append(np.divide(np.sum(arr, axis=0), 3))
    for i in range(len(averaged_arrays)):
        print(compute_deficiency(averaged_arrays[0], averaged_arrays[i]))


if __name__ == "__main__":
    main()
