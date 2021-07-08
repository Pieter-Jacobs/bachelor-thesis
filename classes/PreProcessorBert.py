import numpy as np
from imports import *
from classes.PreProcessor import PreProcessor


class PreProcessorBert(PreProcessor):
    """ 
    Defines an active learning module that is used to take care of the active learning loop
    Attributes:
    ----------
    bertTokenizer: transformers.BertTokenizer 
        The tokenizer used to tokenize sentences for BERT
    temp_token_ids: list
        The placeholder for the input ids to be used for BERT so that they can be used for masking

    Methods
    -------
    id_tokenizer(text):
        Creates token ids for the first sentence from the used texts
    mask_tokenizer():
        Creates the masks for the input ids of the sentences to tell the model about the padding
    create_fields():
        Overrides the method from PreProcessor and adds fields for the masks and the BERT input ids

    Parent:
    """
    __doc__ += PreProcessor.__doc__

    def __init__(self):
        super().__init__()
        self.bertTokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased', do_lower_case=True)
        self.temp_token_ids = None

    def id_tokenizer(self, text):
        """
        Creates token ids for the first sentence from the used texts

        Parameters:
        -----------
        text: str
            The text to be tokenized
        
        Returns:
        --------
        token_ids: list
            The token ids to be used by BERT for classification
        """
        MAX_LENGTH = 50
        sentence = self.split_sentences(text)
        token_ids = self.bertTokenizer.encode(sentence,
                                              truncation=True,
                                              max_length=MAX_LENGTH,
                                              add_special_tokens=True,
                                              return_tensors='pt')
        token_ids = torch.cat([token_ids.squeeze(), token_ids.new_zeros(
            MAX_LENGTH - token_ids.squeeze().size(0))], 0)  # Apply padding
        self.temp_token_ids = token_ids
        return token_ids

    def mask_tokenizer(self, text):
        """
        Creates the masks for the input ids of the sentences to tell the model about the padding
        Parameters:
        -----------
        text: str
            Placeholder value for the text, needed because of the way torch deals with tokenizers

        Returns: 
        --------
        torch.tensor(attention_masks): torch.Tensor
            The attention masks corresponding to the given token_ids
        """
        attention_masks = [int(token > 0)
                           for token in self.temp_token_ids.squeeze()]
        return torch.tensor(attention_masks)

    def create_fields(self):
        """
        Overrides the method from PreProcessor and adds fields for the masks and the BERT input ids

        Returns: 
        --------
        np.concatenate([super().create_fields(), [('token_ids', TOKEN_IDS), ('mask', MASK)]]): list
            The fields to be used for creating examples when using BERT
        """
        TOKEN_IDS = data.Field(sequential=True,
                               batch_first=True,
                               tokenize=self.id_tokenizer,
                               use_vocab=False)
        MASK = data.Field(sequential=True,
                          batch_first=True,
                          tokenize=self.mask_tokenizer,
                          use_vocab=False)
        return np.concatenate(
            [super().create_fields(), [('token_ids', TOKEN_IDS), ('mask', MASK)]])
