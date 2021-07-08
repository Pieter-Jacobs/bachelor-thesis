from imports import *
from classes.Querier import Querier
import numpy as np


@njit
def cosine_similarity(v1, v2):
    """Compute the cosine similarity between vectors v1 and v2"""
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]
        y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    return sumxy/((sumxx**0.5)*(sumyy**0.5))


class QuerierSUD(Querier):
    """ 
    Defines a class that is used query data based on the SUD heuristic

    Attributes:
    -----------
    K = int
        Amount of datapoints with which to compute the density
    Methods
    -------
    compute_similarities(embeddings):
        Computes the cosine similarities between all examples and returns matrix containing them
    compute_densities():
        Computes the density of all examples and returns a list with all densities
    sort_dataset():
        Overrides a Querier method and sorts the dataset based on the examples density*uncertainty

    Parent:
    """
    __doc__ += Querier.__doc__

    def __init__(self, model, iterator, query_function, T, Q, pool_size, K) -> None:
        """
        Parameters:
        -----------
        model: torch.nn 
            The language model used for classification
        query_function: str
            String representing the chosen query function
        pool_size: int
            The pool size used for the redundancy pools in the heuristics
        iterator: torchtext.data.Iterator
            The iterator used to iterate through the unlabeled dataset for computing uncertainties
        uncertainty_matrix: list
            List of the uncertainties of all examples
        Q: int
            The number of datapoints to be queried
        T: int
            The number of stochastic forward passes to be made when computing uncertainties
        K = int
            Amount of datapoints with which to compute the density
        """
        super().__init__(model=model,
                         iterator=iterator,
                         query_function=query_function,
                         T=T,
                         Q=Q,
                         pool_size=pool_size)
        self.K = K

    @staticmethod
    @njit
    def compute_similarities(embeddings):
        """Computes the cosine similarities between all examples and returns matrix containing them"""
        n = len(embeddings)
        distance_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                distance_matrix[i, j] = cosine_similarity(
                    embeddings[i], embeddings[j])
        return distance_matrix

    def compute_densities(self):
        """Computes the density of all examples and returns a list with all densities"""
        embeddings = np.array(
            [ex.embedding for ex in self.iterator.dataset.examples])
        similarities = self.compute_similarities(embeddings)

        densities = np.zeros(len(self.iterator.dataset.examples))
        for i in range(len(similarities)):
            similarities[i] = sorted(similarities[i], reverse=True)
            densities[i] = sum(similarities[i][0:self.K]) / \
                len(similarities[i][0:self.K])
        return densities

    def sort_dataset(self):
        """Overrides Querier method and sorts the dataset based on the examples density*uncertainty"""
        densities = self.compute_densities()
        sud_values = self.uncertainty_matrix * densities
        sorted_idx = np.argsort(
            -sud_values)
        self.iterator.dataset.examples = self.iterator.dataset.examples.reindex(
            sorted_idx)
        self.iterator.dataset.examples.index = [
            i for i in range(len(self.iterator.dataset.examples))]
