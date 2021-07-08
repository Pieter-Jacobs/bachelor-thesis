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


class QuerierRECT(Querier):
    """ 
    Defines a class that is used query and label data based on the RECT heuristic

    Attributes:
    -----------
    l: float
        Threshold by which to select and reject examples from the redundancy pool based on their cosine similarity

    Methods
    -------
    query():
        Overrides a Querier method to query based on the RECT heuristic
    needs_querying():
        Checks whether an example is fit to be queried based on its similarity to examples in the query pool

    Parent:
    """
    __doc__ += Querier.__doc__

    def __init__(self, model, iterator, query_function, T, Q, pool_size, l) -> None:
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
        l = float
            Threshold by which to select and reject examples from the redundancy pool based on their cosine similarity
        """
        super().__init__(model=model,
                         iterator=iterator,
                         query_function=query_function,
                         T=T,
                         Q=Q,
                         pool_size=pool_size)
        self.l = l

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

    def query(self):
        """
        Overrides a Querier method to query based on the RECT heuristic 
        Queries and labels the Q examples out of the redundancy pool which are most dissimilar based on cosine similarity

        Returns:
        --------
        labeled_examples: list
            Examples that were queried and now labeled that are to be added to the labeled training dataset
        """
        labeled_examples = np.array([])
        selected_idx = []
        queried_amount = 0
        embeddings = np.array([ex.embedding for ex in np.array(
            self.iterator.dataset.examples.loc[0:self.pool_size-1])])
        queried_distances = self.compute_similarities(embeddings)
        l = self.l
        while(queried_amount < self.Q):
            for i in range(self.pool_size):
                if i not in selected_idx and self.needs_querying(i, queried_distances, selected_idx, l) and queried_amount < self.Q:
                    self.iterator.dataset.examples.loc[i].label = self.iterator.dataset.examples.loc[i].oracle_label
                    labeled_examples = np.append(
                        labeled_examples, self.iterator.dataset.examples.loc[i])
                    queried_amount += 1
                    selected_idx.append(i)
            l += 0.1    # Increase the threshold because not enough examples are in the query pool yet
        self.iterator.dataset.examples.drop(selected_idx, inplace=True)
        self.iterator.dataset.examples.index = (
            [i for i in range(len(self.iterator.dataset.examples))])
        return labeled_examples

    def needs_querying(self, idx, distance_matrix, selected_idx, l):
        """
        Checks whether an example is fit to be queried based on its similarity to examples in the query pool

        Parameters:
        -----------
        idx: int
            Index of the example that is checked
        distance_matrix: list
            Matrix containing all cosine similarities between examples
        selected_idx: list
            Indexes of examples in the query pool
        l: float
            The current threshold by which to select and reject examples from the redundancy pool based on their cosine similarity
        """
        for i in range(len(distance_matrix)):
            if i in selected_idx and distance_matrix[idx][i] > l and i != idx:
                return False
        return True
