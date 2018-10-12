# -*- coding: utf-8 -*-

import numpy as np
from gordo_components.dataset.base import GordoBaseDataset


class RandomDataset(GordoBaseDataset):
    """
    Get a GordoBaseDataset which returns random values for X and y based
    """
    def __init__(self, size=100, n_features=20, **kwargs):
        self.size = size
        self.n_features = n_features

    def get_train(self):
        """return X and y data"""
        X = np.random.random(
            size=self.size * self.n_features).reshape(-1, self.n_features)
        y = np.random.randint(self.size)
        return X, y

    def get_test(self):
        X = np.random.random(
            size=self.size * self.n_features).reshape(-1, self.n_features)
        return X
