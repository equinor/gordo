# -*- coding: utf-8 -*-

import abc


class GordoBaseDataset:

    @abc.abstractmethod
    def __init__(self, **kwargs):
        """
        Initialize the dataset.
        """

    @abc.abstractmethod
    def get_train(self):
        """
        Using initialized params, returns X, y as numpy arrays.
        """

    @abc.abstractmethod
    def get_test(self):
        """
        Using initialized params, returns X as a numpy array
        """
