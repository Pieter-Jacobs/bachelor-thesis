from imports import *
from classes.SubscriptableExample import SubscriptableExample


class DataFrameDataset(data.Dataset):
    """
    Defines a type of dataset that creates a pytorch dataset object from a pandas dataframe

    Arguments:
    ----------
    fields: list 
        The fields that are to be assigned to an example
    df: pandas.DataFrame
        The dataframe that is to be transformed into a dataset 
        Default is None
    Examples: list
        The examples to be used in the dataset, mainly meant for loading a dataset
        Default is []
    is_unlabeled: bool
        Indicates whether the used dataframe is to be used as the unlabeled dataset
        Default is False

    Methods
    -------
    createExample(row, fields, is_unlabeled):
        Overwrites method from the pytorch dataset class. Creates examples with the needed fields from dataframe rows.
    sort_key(ex):
        Makes sure the data is sorted based on text length
    to_csv(file_name):
        Writes the dataset to a csv file

    Parent:
    """
    __doc__ += data.Dataset.__doc__

    def __init__(
            self,
            fields,
            df=None,
            examples=[],
            is_unlabeled=False,
            **kwargs):
        """
        Parameters:
        ----------
        fields: list 
            The fields that are to be assigned to an example
        df: pandas.DataFrame
            The dataframe that is to be transformed into a dataset 
            Default is None
        Examples: list
            The examples to be used in the dataset, mainly meant for loading a dataset
            Default is []
        is_unlabeled: bool
            Indicates whether the used dataframe is to be used as the unlabeled dataset
            Default is False
        """
        if len(examples) == 0:  
            examples = df.apply(
                lambda row: self.createExample(
                    row, fields, is_unlabeled), axis=1)
            super().__init__(examples, fields, **kwargs)
        else:   # If examples were provided, we can simply create a dataset with those examples
            super().__init__(examples, fields)

    def createExample(cls, row, fields, is_unlabeled):
        """
        Overwrites method from the pytorch dataset class. Creates examples with the needed fields from dataframe rows
        
        Parameters:
        ----------
        row: pandas.Series
            row containing all column values for a specific datapoint
        fields: list 
            The fields that are to be assigned to an example
        is_unlabeled: bool
            Indicates whether the used dataframe is to be used as the unlabeled dataset
        """
        if is_unlabeled:
            # Assign a placeholder label to 'unlabeled data' to make it iterable for BucketIterator
            row['label'] = next(iter(fields[1][1].vocab.stoi))
        return SubscriptableExample.fromlist(
            [
                row['text'],
                row['label'],
                row['oracle_label'],
                row['text'],
                row['text'],
                row['text']],
            fields)

    @staticmethod
    def sort_key(ex):
        """Makes sure the data is sorted based on text length"""
        return len(ex['text'])

    def to_csv(self, file_name):
        """Writes the dataset to a csv file based on an input filename"""
        texts = [ex.text for ex in self.examples]
        labels = [ex.label for ex in self.examples]
        pd.DataFrame(data={'Text': texts, 'Label': labels}).to_csv(file_name)
