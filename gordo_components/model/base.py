# -*- coding: utf-8 -*-

import abc


class GordoBaseModel:

    _model = None

    @abc.abstractproperty
    def model(self):
        """Return instance of the underlying predictor model"""
        return self._model

    @abc.abstractmethod
    def __init__(self, n_features=None, **kwargs):
        """Return model from a config"""

    @abc.abstractmethod
    def fit(self, X, y):
        """Fit the model AND return self"""

    @abc.abstractclassmethod
    def predict(self, X):
        """make a prediction given X"""
    