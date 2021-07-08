import numpy as np
from imports import *
from classes.DataFrameDataset import DataFrameDataset
from classes.StringField import StringField


class PreProcessor:
    """ 
    Defines a PreProcessor that is used to take reprocess all data into a general format

    Methods
    -------
    pre_process(df, selected_columns, remove_labels):
        Preprocesses the dataframe into a generalised format
    rename_columns(df, columns):
        Generalises the header names of the selected columns to 'text' and 'label'
    drop_columns(df, columns):
        Drops the given columns from the given dataframe
    remove_non_ascii(text):
        Removes any non printable characters from a given text and returns the cleaned text
    tokenizer(text):
        Tokenizes a given text with nltk and returns the tokens
    remove_empty_strings(text):
        Removes empty strings from the given dataframe and returns the dataframe
    split_sentences(text):
        Splits a given text into sentences and returns the first
    create_fields():
        Creates the necassary fields for a standard classification task
    create_datasets(labeled_df, fields, is_test, unlabeled_df=None):
        Builds datasets from a given dataframe
    """

    def pre_process(self, df, selected_columns, remove_labels):
        """
        Preprocesses the dataframe into a generalised format

        Parameters:
        -----------
        df: pandas.DataFrame
            The dataframe to be preprocessed
        selected_columns: list
            The columns that are to be used for classification
        remove_labels: bool
            Option to tell whether we are dealing with the unlabeled dataset or not

        Returns:
        --------
        df: pandas.DataFrame
            The preprocessed dataframe
        """
        df = self.drop_columns(df, selected_columns)
        df = self.rename_columns(df, selected_columns)
        df['text'] = df['text'].apply(
            lambda text: self.remove_non_ascii(text))
        df = self.remove_empty_strings(df)
        df.insert(2, "oracle_label", df['label'], True)
        if remove_labels:
            df['label'] = df['label'].apply(lambda x: None)
        return df

    def rename_columns(self, df, columns):
        """
        Generalises the header names of the selected columns to 'text' and 'label'

        Parameters:
        -----------
        df: pandas.DataFrame
            The dataframe to be preprocessed
        columns: list
            The columns that are to be renamed

        Returns:
        --------
        df: pandas.DataFrame
            The dataframe with renamed columns
        """
        df.columns = [
            'text',
            'label'] if columns[0] < columns[1] else [
            'label',
            'text']
        return df

    def drop_columns(self, df, columns):
        """
        Drops the given columns from the given dataframe

        Parameters:
        -----------
        df: pandas.DataFrame
            The dataframe to be preprocessed
        columns: list
            The columns that are to be used (not dropped)

        Returns:
        --------
        df: pandas.DataFrame
            The dataframe with dropped columns
        """
        cols_to_drop = [df.columns[col]
                        for col in range(len(df.columns)) if col not in columns]
        df.drop(columns=cols_to_drop, axis=1, inplace=True)
        return df

    def remove_non_ascii(self, text):
        """Removes any non printable characters from a given text and returns the cleaned text"""
        filter(lambda x: x in string.printable, text)
        return text

    def tokenizer(self, text):
        """Tokenizes a given text with nltk and returns the tokens"""
        sentence = self.split_sentences(text)
        return sentence

    def remove_empty_strings(self, df):
        """Removes empty strings from the given dataframe and returns the dataframe"""
        return df.loc[df['text'].str.len() > 0]

    def split_sentences(self, text):
        """Splits a given text into sentences and returns the first"""
        return tokenize.sent_tokenize(text)[0]

    def create_fields(self):
        """
        Creates the necassary fields for a standard classification task

        Returns: 
        --------
        [('text', TEXT), ('label', LABEL), ('oracle_label', ORACLE_LABEL)]: list
            The fields to be used for creating examples
        """
        TEXT = StringField(sequential=True,
                           batch_first=True,
                           tokenize=self.tokenizer,
                           use_vocab=False)
        LABEL = data.LabelField(dtype=torch.float)
        ORACLE_LABEL = data.LabelField(dtype=torch.float)
        return [('text', TEXT), ('label', LABEL),
                ('oracle_label', ORACLE_LABEL)]

    def create_datasets(self, labeled_df, fields, is_test, unlabeled_df=None):
        """
        Builds datasets from a given dataframe

        Parameters:
        ----------
        labeled_df: pandas.DataFrame
            Dataframe to be s
        fields: list 
            The fields that are to be assigned to an example
        is_test: bool
            Indicates whether the used dataframe is a test or validation set
        unlabeled_df=None: pandas.DataFrame
            The dataframe used for data that is yet to be labeled
        Returns:
        --------
        labeled_ds: DataFrameSet
            The labeled dataset that is to be used for evaluation of the active learning rounds
        unlabeled_ds: DataFrameSet
            The dataset containing data that is still to be labeled in the active learning rounds
        """
        unlabeled_ds = None
        labeled_ds = DataFrameDataset(fields, labeled_df)
        if not is_test:
            fields[1][1].build_vocab(labeled_ds)
            fields[2][1].build_vocab(labeled_ds)
            unlabeled_ds = DataFrameDataset(
                fields, unlabeled_df, is_unlabeled=True)
        return labeled_ds, unlabeled_ds
