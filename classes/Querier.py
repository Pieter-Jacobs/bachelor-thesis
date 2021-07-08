from imports import *
import numpy as np


class Querier:
    """ 
    Defines a class that is used query data based on computed uncertainties

    Attributes:
    -----------
    dispatcher: dict
        Dispatcher used to dispatch the choice of query function
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

    Methods
    -------
    sort_dataset():
        Sorts the dataset based on computed uncertainties
    assign_uncertainties():
        Loop through the dataset and assign the uncertainties to the examples through use of Monte Carlo Dropout
    query():
        Queries and labels the Q examples with the highest uncertainty
    choose_query_function(query_function):
        Makes sure the user inputs a valid query function and returns the chosen query function as callback
    variation_ratio(pred_matrix):
        Computes and returns the variation ratios of the predictions in the given prediction matrix
    predictive_entropy(pred_matrix):
        Computes and returns predictive entropies of the predictions in the given prediction matrix
    mutual_information(pred_matrix):
        Computes and returns BALD of the predictions in the given prediction matrix
    """

    def __init__(self, model, query_function, iterator, Q, T, pool_size=None):
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
        """
        self.dispatcher = {
            'variation_ratio': self.variation_ratio,
            'predictive_entropy': self.predictive_entropy,
            'mutual_information': self.mutual_information
        }
        self.model = model
        self.query_function = self.choose_query_function(query_function)
        self.pool_size = pool_size if pool_size is not None else Q
        self.iterator = iterator
        self.uncertainty_matrix = []
        self.Q = Q
        self.T = T

    def sort_dataset(self):
        """Sorts the dataset based on computed uncertainties"""
        sorted_idx = np.argsort(
            -self.uncertainty_matrix)
        self.iterator.dataset.examples = self.iterator.dataset.examples.reindex(
            sorted_idx)
        self.iterator.dataset.examples.index = [
            i for i in range(len(self.iterator.dataset.examples))]

    def assign_uncertainties(self):
        """Loop through the dataset and assign the uncertainties to the examples through use of Monte Carlo Dropout"""
        self.uncertainty_matrix = np.array([])
        self.model.eval()

        for m in self.model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.training = True   # Turn on dropout

        with torch.no_grad():
            torch.cuda.empty_cache()
            for batch in self.iterator:
                preds = np.array([self.model(batch.token_ids,
                                             token_type_ids=None,
                                             attention_mask=batch.mask
                                             )[0].cpu().numpy() for sample in range(int(self.T))])
                preds = np.transpose(preds, (1, 0, 2))
                preds = softmax(preds, axis=2)
                self.uncertainty_matrix = np.append(
                    self.uncertainty_matrix, self.query_function(preds))

    def query(self):
        """
        Queries and labels the Q examples with the highest uncertainty

        Returns:
        --------
        labeled_examples: list
            Examples that were queried and now labeled that are to be added to the labeled training dataset
        """
        labeled_examples = np.array([])
        for i in range(self.Q):
            if len(self.iterator.dataset.examples) > 0:
                self.iterator.dataset.examples.loc[i].label = self.iterator.dataset.examples.loc[i].oracle_label
                labeled_examples = np.append(
                    labeled_examples, self.iterator.dataset.examples.loc[i])
                self.iterator.dataset.examples.drop(i, inplace=True)
        self.iterator.dataset.examples.index = (
            [i for i in range(len(self.iterator.dataset.examples))])
        return labeled_examples

    def choose_query_function(self, query_function):
        """Makes sure the user inputs a valid query function and returns the chosen query function as callback"""
        while query_function not in self.dispatcher.keys():
            query_function = input(
                '''invalid algorihm choice, try one of the following strings:
                variation_ratio
                predictive_entropy
                mutual_information\n''')
        return self.dispatcher[query_function]

    def variation_ratio(self, pred_matrix):
        """Computes and returns the variation ratios of the predictions in the given prediction matrix"""
        preds = [preds.argmax(1) for preds in pred_matrix]
        mode = stats.mode(preds, axis=1)
        return 1 - (mode[1].squeeze() / self.T)

    @staticmethod
    @njit
    def predictive_entropy(pred_matrix):
        """Computes and returns predictive entropies of the predictions in the given prediction matrix"""
        n = len(pred_matrix)
        uncertainty_matrix = np.zeros(n)
        inner_dim = pred_matrix.shape[len(pred_matrix.shape) - 1]
        for i, preds in enumerate(pred_matrix):
            sum_ = np.zeros(inner_dim)
            for pred in preds:
                pred = np.add(  # prevent(log(0))
                    pred,
                    1e-17)
                sum_ = np.add(sum_, pred)
            avg_preds = np.divide(sum_, len(preds))
            uncertainty_matrix[i] = -1 * \
                np.sum(np.multiply(avg_preds, np.log2(avg_preds)))
        return uncertainty_matrix

    def mutual_information(self, pred_matrix):
        """Computes and returns BALD of the predictions in the given prediction matrix"""
        uncertainty_matrix = np.array([])
        inner_dim = pred_matrix.shape[len(pred_matrix.shape) - 1]
        for preds in pred_matrix:
            sum_ = np.zeros(inner_dim)
            for pred in preds:
                pred = np.add(  # prevent log(0)
                    pred,
                    1e-17)
                sum_ = np.add(sum_, np.multiply(pred, np.log2(pred)))
            uncertainty_matrix = np.append(
                uncertainty_matrix, np.sum(sum_ / self.T))
        return np.add(self.predictive_entropy(pred_matrix), uncertainty_matrix)
