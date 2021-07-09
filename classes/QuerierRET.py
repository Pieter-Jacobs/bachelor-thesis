from imports import *
from classes.Querier import Querier
from classes.DataFrameDataset import DataFrameDataset
import numpy as np


class QuerierRET(Querier):
    """ 
    Defines a class that is used query and label data based on the RET heuristic

    Attributes:
    -----------
    trainer: Trainer
        Trainer used to retrain on the redundancy pool

    Methods
    -------
    query():
        Overrides a Querier method to query based on the RET heuristic
    assign_pool_uncertainties():
        Computes and returns the uncertainties of the redundancy pool
        
    Parent:
    """
    __doc__ += Querier.__doc__

    def __init__(self, model, iterator, query_function, T, Q, trainer, pool_size) -> None:
        super().__init__(model=model,
                         iterator=iterator,
                         query_function=query_function,
                         T=T,
                         Q=Q,
                         pool_size=pool_size)
        self.trainer = trainer

    def query(self):
        """
        Overrides a Querier method to query based on the RETT heuristic 
        Queries and labels the Q examples out of the redundancy pool which are most dissimilar based on constantly recomputed uncertainties

        Returns:
        --------
        uncertainties: list
            Examples that were queried and now labeled that are to be added to the labeled training dataset
        """
        labeled_examples = np.array([])
        queried_amount = 0
        # Prevent an error from occuring when there is not enough data to fill RP
        if len(self.iterator.dataset.examples) < self.pool_size:
            self.pool_size = len(self.iterator.dataset.examples)
        uncertainty_rp = sorted(self.uncertainty_matrix, reverse=True)[
            0:self.pool_size]
        initial_dataset = self.trainer.train_iterator.dataset
        while queried_amount < self.Q and len(self.iterator.dataset.examples) > 0:
            idx = np.argmax(uncertainty_rp)
            self.iterator.dataset.examples.loc[idx].label = self.iterator.dataset.examples.loc[idx].oracle_label
            labeled_examples = np.append(
                labeled_examples, self.iterator.dataset.examples.loc[idx])
            self.trainer.train_iterator.dataset = DataFrameDataset(
                examples=[self.iterator.dataset.examples.loc[idx]], fields=self.iterator.dataset.fields)
            self.trainer.training_step()    # Retrain on top example
            self.iterator.dataset.examples.drop(idx, inplace=True)
            self.iterator.dataset.examples.index = (
                [i for i in range(len(self.iterator.dataset.examples))])
            if(len(self.iterator.dataset.examples) > 0):
                uncertainty_rp = self.assign_pool_uncertainties(
                    self.iterator.dataset.examples[0:(self.pool_size-len(labeled_examples))])   # Recompute uncertainties for RP
            queried_amount += 1
        self.trainer.train_iterator.dataset = initial_dataset
        return labeled_examples

    def assign_pool_uncertainties(self, examples):
        """
        Computes and returns the uncertainties of the redundancy pool
        
        Parameters:
        -----------
        examples:
            examples in the redundancy pool to compute uncertainties for
            
        Returns:
        -------- 
        uncertainty_matrix: list
            uncertainties of all examples in the redundancy pool after retraining on the top example
        """
        uncertainty_matrix = []
        iterator = data.BucketIterator(
            dataset=DataFrameDataset(
                examples=examples, fields=self.iterator.dataset.fields),
            batch_size=self.pool_size,
            device=torch.device('cuda'),
            sort_within_batch=False,
            sort=False,
            shuffle=False
        )

        self.model.eval()
        for m in self.model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.training = True   # Turn on dropout

        with torch.no_grad():
            torch.cuda.empty_cache()
            for batch in iterator:
                preds = np.array([self.model(batch.token_ids,
                                             token_type_ids=None,
                                             attention_mask=batch.mask
                                             )[0].cpu().numpy() for sample in range(int(self.T))])
                preds = np.transpose(preds, (1, 0, 2))
                preds = softmax(preds, axis=2)

                uncertainties = self.query_function(preds)
                # Prevent an error occuring when just one example is in the batch
                if not isinstance(uncertainties, np.ndarray):
                    uncertainties = np.array([uncertainties])
                uncertainty_matrix = np.concatenate(
                    (uncertainty_matrix, uncertainties))
        return uncertainty_matrix
