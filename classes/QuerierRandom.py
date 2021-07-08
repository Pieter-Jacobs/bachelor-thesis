from imports import *
from classes.Querier import Querier


class QuerierRandom(Querier):
    """ 
    Defines a class that is used to query based on random sampling
    
    Parent:
    """
    __doc__ += Querier.__doc__

    def sort_dataset(self):
        """Overrides a Queried method to sort the dataset randomly"""
        random.Random(1815).shuffle(self.iterator.dataset.examples)

    def assign_uncertainties(self):
        """Overrides a Querier method to ignore it alltogether as uncertainties don't need to be computed"""
        pass
