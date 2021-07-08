from imports import *


class StringField(data.Field):
    """
    Defines a Field to deal with strings

    Methods: 
    --------
    process(batch, device=None):
        Pads the input batch and returns it
    
    Parent:
    """
    __doc__ += data.Field.__doc__

    def process(self, batch, device=None):
        """Pads the input batch and returns it"""
        padded = self.pad(batch)
        return padded
