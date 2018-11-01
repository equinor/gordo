# -*- coding: utf-8 -*-

import abc
from sklearn.base import BaseEstimator


class GordoBaseModel(BaseEstimator):

    """The underlying model, ie the raw Keras, MXNet, PyTorch, ect model"""
    model = None

    @abc.abstractmethod
    def __init__(self, n_features=None, **kwargs):
        """Initialize the model"""

    @abc.abstractmethod
    def fit(self, X, y=None):
        """Fit the model AND return self"""

    @abc.abstractmethod
    def predict(self, X):
        """make a prediction given X"""

    @abc.abstractmethod 
    def get_params(deep=False):
        """Return a dict containing all parameters used to initialized object"""
