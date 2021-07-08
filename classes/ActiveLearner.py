from imports import *
import numpy as np


class ActiveLearner():
    """ 
    Defines an active learning module that is used to take care of the active learning loop
    Attributes:
    ----------
    model: torch.nn 
        The language model used for classification
    trainer: Trainer
        The trainer object used to train the model
    querier: Querier
        The querier object used to query and annotate
    optimizer_config: dict
        The configuration settings containing the parameters for tbe Adam optimizer
    metric_file: str
        The file to write the evaluation metrics to
    Methods
    -------
    loop():
        Makes active learning steps until all data is labeled.
    step():
        Steps through one active learning round:
        1: Retrain the model.
        2: Annotate the most informative data.
        3: Append the annotated data to the training set.
    """

    def __init__(self, model, trainer, querier, optimizer_config, metric_file):
        """ 
        Parameters:
        ----------
        model: Model 
            the language model used for classification
        trainer: Trainer
            the trainer object used to train the model
        querier: Querier
            the querier object used to query and annotate
        optimizer_config: dict
            the configuration settings containing the parameters for tbe Adam optimizer
        metric_file: str
            The file to write the evaluation metrics to
        """
        self.model = model
        self.trainer = trainer
        self.querier = querier
        self.optimizer_config = optimizer_config
        self.metric_file = metric_file

    def loop(self):
        """Makes active learning steps until all data is labeled"""
        while len(self.querier.iterator.dataset.examples) > 0:
            self.step()
        self.trainer.reset_parameters(self.optimizer_config)
        self.trainer.train_early_stopping(self.optimizer_config)
        f = open(os.path.join(hydra.utils.get_original_cwd(),
                 self.metric_file + '_accuracy.txt'), "a")
        f.write('\n')
        f.close()
        f = open(os.path.join(hydra.utils.get_original_cwd(),
                 self.metric_file + '_roc.txt'), "a")
        f.write('\n')
        f.close()

        del self.model  # Delete the model to clear memory

    def step(self):
        """
        Steps through one active learning round:
            1: Retrain the model.
            2: Annotate the most informative data.
            3: Append the annotated data to the training set.
        """
        self.trainer.reset_parameters(self.optimizer_config)
        self.trainer.train_early_stopping(self.optimizer_config)
        self.querier.assign_uncertainties()
        self.querier.sort_dataset()
        self.trainer.train_iterator.dataset.examples = np.append(
            self.trainer.train_iterator.dataset.examples, self.querier.query())
