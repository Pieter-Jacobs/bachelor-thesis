import json
from functools import reduce
import warnings
import torchtext.data as data


class SubscriptableExample(data.Example):
    """
    Makes the Example class subscriptable by adding __getitem__
    """

    def __getitem__(self, x):
        return getattr(self, x)
