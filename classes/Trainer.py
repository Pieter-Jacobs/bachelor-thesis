from imports import *
import numpy as np


class Trainer():
    """ 
    Defines an class meant for training a model
    Attributes:
    ----------
    model: torch.nn 
        The language model used for classification
    train_iterator: torchtext.data.Iterator
        The iterator used to batch and iterate through the data
    val_iterator: torchtext.data.Iterator
        The iterator used to batch and iterate through the data
    criterion: torch.nn
        The loss function used to train the model with
    optimizer: transformers.AdamW
        The Adam Optimizer used for optimization of the training process
    evaluator: Evaluator
        The Evaluator used for computing the validation loss in early stopping
    n_epochs: int
        Number of epochs to train the model for

    Methods
    -------
    train_early_stopping(config):
        Trains the model for the chosen amount of epochs and chooses the model with the lowest validation loss
    validation_step():
        Evaluates the performance of the model on the validation set
    training_step():
        Trains the model based on one pass through all data
    compute_accuracy(preds, y):
        Computes the accuracy of the made predictions, so it can be printed
    reset_parameters(config):
        Resets the model its parameters to its initial ones
    """

    def __init__(self, model, train_iterator, criterion, optimizer, n_epochs, evaluator, val_iterator=None):
        """
        Parameters:
        -----------
        model: torch.nn 
            The language model used for classification
        train_iterator: torchtext.data.Iterator
            The iterator used to batch and iterate through the data
        val_iterator: torchtext.data.Iterator
            The iterator used to batch and iterate through the data
        criterion: torch.nn
            The loss function used to train the model with
        optimizer: transformers.AdamW
            The Adam Optimizer used for optimization of the training process
        evaluator: Evaluator
            The Evaluator used for computing the validation loss in early stopping
        n_epochs: int
            Number of epochs to train the model for
        """
        self.model = model
        self.train_iterator = train_iterator
        self.val_iterator = val_iterator
        self.criterion = criterion
        self.optimizer = optimizer
        self.evaluator = evaluator
        self.n_epochs = n_epochs

    def train_early_stopping(self, config):
        """
        Trains the model for the chosen amount of epochs and chooses the model with the lowest validation loss

        Parameters:
        -----------
        config: dict
            Configuration dictionary which contains the chosen hyperparameters for the optimizer
        """
        loss = []
        acc = []
        epoch = 0
        val_losses = np.array([])
        print("Starting training...")
        for epoch in range(self.n_epochs):
            torch.save(self.model.state_dict(), os.path.join(
                hydra.utils.get_original_cwd(), "saves" + os.path.sep + 'model-early-stopping' + str(epoch) + '.pkl'))
            print("Epoch number: " + str(epoch))
            train_acc, train_loss = self.training_step()
            print(
                f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
            loss.append(train_loss)
            acc.append(train_acc)
            val_acc, val_loss = self.validation_step()
            print(
                f'Val Loss: {val_loss:.3f} | Val Acc: {val_acc*100:.2f}%')
            val_losses = np.append(val_losses, val_loss)
        self.model.load_state_dict(torch.load(os.path.join(
            hydra.utils.get_original_cwd(), "saves" + os.path.sep + "model-early-stopping" + str(np.argmin(val_losses)) + ".pkl")), strict=False)
        self.optimizer.params = AdamW(
            self.model.parameters(), lr=config.lr, eps=config.eps)
        self.evaluator.evaluate()

    def validation_step(self):
        """"
        Evaluates the performance of the model on the validation set
        
        Returns:
        (epoch_acc / len(self.val_iterator)): float
            Average validation accuracy over the different batches
        (epoch_loss / len(self.val_iterator)): float
            Average validation loss over the different batches
        """
        epoch_acc = 0
        epoch_loss = 0
        self.model.eval()
        with torch.no_grad():
            for batch in self.val_iterator:
                predictions = self.model(batch.token_ids,
                                         token_type_ids=None,
                                         attention_mask=batch.mask,
                                         labels=batch.label.long())
                loss = predictions[0]
                acc = self.compute_accuracy(predictions[1], batch.label.long())
                epoch_acc += acc
                epoch_loss += float(loss.item())
        return (epoch_acc / len(self.val_iterator)), (epoch_loss / len(self.val_iterator))

    def training_step(self):
        """
        Trains the model based on one pass through all data

        Returns:
        --------
        epoch_acc / len(self.train_iterator): float 
            Average training accuracy over the different batches
        epoch_loss / len(self.train_iterator): float
            Average training loss over the different batches
        """
        epoch_loss = 0
        epoch_acc = 0
        self.model.train()
        for batch in self.train_iterator:
            self.optimizer.zero_grad()
            predictions = self.model(batch.token_ids,
                                     token_type_ids=None,
                                     attention_mask=batch.mask,
                                     labels=batch.label.long())
            loss = predictions[0]
            acc = self.compute_accuracy(predictions[1], batch.label.long())
            loss.backward()
            self.optimizer.step()
            epoch_loss += float(loss.item())
            epoch_acc += float(acc)
        return epoch_acc / len(self.train_iterator), epoch_loss / len(self.train_iterator)

    def compute_accuracy(self, preds, y):
        """Computes the accuracy of the made predictions, so it can be printed"""
        correct = 0.0
        for i, pred in enumerate(preds):
            # Check if prediction is the same as supervised label
            if torch.tensor(np.argmax(pred.detach().cpu().numpy())) == y[i].cpu():
                correct += 1
        return correct / len(preds)

    def reset_parameters(self, config):
        """
        Resets the model its parameters to its initial ones
        
        Parameters:
        -----------
        config: dict
            Configuration dictionary which contains the chosen hyperparameters for the optimizer """
        self.model.load_state_dict(torch.load(os.path.join(
            hydra.utils.get_original_cwd(), "saves" + os.path.sep + "model.pkl")), strict=False)
        self.optimizer = AdamW(
            self.model.parameters(), lr=config.lr, eps=config.eps)
