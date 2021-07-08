from imports import *
from classes.PreProcessorBert import PreProcessorBert
import numpy as np


class PreProcessorSBert(PreProcessorBert):
    """ 
    Defines a preprocessor that is used to process all data into a general format to be used by the BERT model
    and to also assign SentenceBERT embeddings to all texts

    Attributes:
    -----------
    sentenceTransformer: sentence_transformers.SentenceTransformer
        The SentenceBERT model used for computing the sentence embeddings

    Methods
    -------
    emb_tokenizer(text):
        Computes the sentence embedding for a text and returns that embedding
    mask_tokenizer():
        Creates the masks for the input ids of the sentences to tell the model about the padding
    create_fields():
        Overrides the method from PreProcessor and adds fields for the masks and the BERT input ids

    Parent:
    """
    __doc__ += PreProcessorBert.__doc__

    def __init__(self):
        super().__init__()
        self.sentenceTransformer = SentenceTransformer(
            'paraphrase-distilroberta-base-v1')

    
    def emb_tokenizer(self, text):
        """Computes the sentence embedding for a text and returns that embedding"""
        sentence = self.split_sentences(text)
        embedding = self.sentenceTransformer.encode(
            sentence, show_progress_bar=False)
        return embedding

    def create_fields(self):
        """
        Overrides the method from PreProcessorBert and adds a field for the sentence embeddings

        Returns: 
        --------
        [super().create_fields(), [('embedding', EMBEDDING)]]): list
            The fields to be used for creating examples when making use of embeddings
        """
        EMBEDDING = data.Field(sequential=True,
                               batch_first=True,
                               tokenize=self.emb_tokenizer,
                               use_vocab=False,
                               dtype=torch.float
                               )
        return np.concatenate(
            [super().create_fields(), [('embedding', EMBEDDING)]])
