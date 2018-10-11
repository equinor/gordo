# -*- coding: utf-8 -*-

import numpy as np

class Dataset:

    # TODO: Implement me

    @classmethod
    def from_config(cls, data_config):
        return Dataset()

    def get_train(self):
        """return X and y data"""
        X = np.random.random(size=100).reshape(-1, 20)
        y = np.random.randint(5)
        return X, y

    def get_test(self):
        X = np.random.random(size=100).reshape(-1, 20)
        return X
