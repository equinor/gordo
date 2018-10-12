#  -*- coding: utf-8 

import os
try:
    import cPickle as pickle 
except ImportError:
    import pickle
import joblib

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
    try:
        print(os.listdir(os.path.dirname(path)))
        print('Model path: {}'.format(path))
    except:
        print('Path "{}" is not a file'.format(path))

    if path.endswith('.h5'):
        from keras.models import Model
        _model = Model.load_model(path)
        model = joblib.load(path.replace('.h5', '.pkl'))
        model._model = _model
        return model

    with open(path, 'rb') as file:
        return joblib.load(file)
    