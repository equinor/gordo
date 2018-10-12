# -*- coding: utf-8 -*-

import abc


class GordoBaseDataset:

    @abc.abstractmethod
    def __init__(self, **kwargs):
        """
        Initialize the dataset.
        """

    @abc.abstractmethod
    def get_data(self):
        """
        Using initialized params, returns X, y as numpy arrays.
        """
