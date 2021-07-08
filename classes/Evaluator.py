from imports import *
import numpy as np


class Evaluator:
    """ 
    Defines an active learning module that is used to take care of the active learning loop
    Attributes:
    ----------
    model: torch.nn 
        The language model used for classification
    iterator: torchtext.data.Iterator
        The iterator used to batch and iterate through the data
    n_epochs: int
        The amount of epochs to evaluate the dataset for
    feature_size: int
        The amount of features of the used dataset

    Methods
    -------
    evaluate():
        Makes evaluation steps corresponding to the amount of epochs and prints the loss and accuracy
    evaluation_step():
        Evaluates the performance of the model on the labeled data
    compute_metrics(preds, y):
        Computes the accuracy and the ROCAUC score of the made predictions
    generate_one_hot(label):
        Creates a one hot vector from a given label index
    """

    def __init__(self, model, iterator, n_epochs, feature_size, metric_file):
        """
        Parameters:
        ----------
        model: torch.nn
            The language model used for classification
        iterator: torchtext.data.Iterator
            The iterator used to batch and iterate through the data
        n_epochs: int
            The amount of epochs to evaluate the dataset for
        feature_size: int
            The amount of features of the used dataset
        metric_file: str
            The file to write the evaluation metrics to
        """
        self.model = model
        self.iterator = iterator
        self.n_epochs = n_epochs
        self.feature_size = feature_size
        self.metric_file = metric_file

    def evaluate(self):
        """
        Makes evaluation steps corresponding to the amount of epochs and prints the loss and accuracy

        Returns:
        avg_loss: float
            Average loss of the model predictions
        """
        acc = []
        loss = []
        print("Starting Evaluating...")
        for epoch in range(self.n_epochs):
            print("Epoch number: " + str(epoch))
            test_acc, test_loss = self.evaluation_step()

            print(f'\t Val Acc: {test_acc*100:.2f}%')
            print(f'\t Val Loss: {test_loss:.2f}')
            acc.append(test_acc)
            loss.append(test_loss)
        avg_loss = sum(loss) / len(loss)
        return avg_loss

    def evaluation_step(self):
        """
        Evaluates based on one pass through all data

        Returns:
         (epoch_acc / len(self.iterator)), (epoch_loss / len(self.iterator)): float
            Average accuracy and loss over the different batches of the epoch
        """
        epoch_acc = 0
        epoch_loss = 0
        auc = 0
        auc_is_undefined = False
        # Tell the model that we are going to evaluate
        self.model.eval()
        # We don't want the gradient to be altered during evaluation, so we
        # ignore it
        with torch.no_grad():
            for batch in self.iterator:
                predictions = self.model(batch.token_ids,
                                         token_type_ids=None,
                                         attention_mask=batch.mask,
                                         labels=batch.label.long())
                loss = predictions[0]
                acc, roc_auc = self.compute_metrics(
                    predictions[1], batch.label.long())
                epoch_acc += acc
                if roc_auc is None:
                    auc = None
                    auc_is_undefined = True
                if not auc_is_undefined:
                    auc += roc_auc
                epoch_loss += float(loss.item())
        f = open(os.path.join(hydra.utils.get_original_cwd(),
                 self.metric_file + '_accuracy.txt'), "a")
        f.write(str(epoch_acc / len(self.iterator)))
        f.write(" ")
        f.close()
        f = open(os.path.join(hydra.utils.get_original_cwd(),
                 self.metric_file + '_roc.txt'), "a")
        if not auc_is_undefined:
            f.write(str(auc / len(self.iterator)))
        else:
            f.write("Undefined")
        f.write(" ")
        f.close()
        return (epoch_acc / len(self.iterator)), (epoch_loss / len(self.iterator))

    def compute_metrics(self, preds, y):
        """
        Computes the accuracy and the ROCAUC score of the made predictions
        If the one of the batches contained one class, the ROCAUC score is ignored
        
        Parameters:
        -----------
        preds: list
            The model softmax classifications
        y: list
            The supervised labels of the examples

        Returns:
        --------
        accuracy_score(labels, pred_labels): float
            The achieved test accuracy
        roc_auc_score(labels=labels, y_true=one_hot, y_score=softmax_preds, multi_class='ovo', average='weighted'): float
            The weighted ROCAUC score for multi class classification
        roc_auc_score(labels, positive_pred): float
            The ROCAUC score for binary classification
        """
        pred_labels = np.array([])
        softmax_preds = np.array([]).reshape(0, self.feature_size)
        labels = np.array([])
        for i, pred in enumerate(preds):
            pred_labels = np.append(
                pred_labels, np.argmax(pred.detach().cpu().numpy()))
            softmax_preds = np.vstack(
                (softmax_preds, softmax(pred.detach().cpu().numpy())))
            labels = np.append(labels, y[i].cpu())
        one_hot = np.array([self.generate_one_hot(label) for label in labels])
        if self.feature_size > 2:
            try:
                return accuracy_score(labels, pred_labels), roc_auc_score(labels=labels, y_true=one_hot, y_score=softmax_preds, multi_class='ovo', average='weighted')
            except:
                return accuracy_score(labels, pred_labels), None
        else:
            positive_pred = np.array([x[1] for x in softmax_preds])
            try:
                return accuracy_score(labels, pred_labels), roc_auc_score(labels, positive_pred)
            except:
                return accuracy_score(labels, pred_labels), None

    def generate_one_hot(self, label):
        """Creates a one hot vector from a given label index"""
        one_hot = np.zeros(self.feature_size)
        one_hot[int(label)] = 1
        return one_hot
