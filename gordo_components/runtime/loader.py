#  -*- coding: utf-8 

import os
try:
    import cPickle as pickle 
except ImportError:
    import pickle

def load_model(path):
    """
    Load a model from a given path, expected to be mounted locally. 

    Paramters:
        path: str - Location of model to load

    Returns:
        Model

    Raises:
        FileNotFoundError
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Model file at '{}' does not exist!".format(path)
        )
    with open(path) as file:
        return pickle.load(file)
    