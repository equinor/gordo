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

    @abc.abstractmethod
    def get_metadata(self):
        """
        Returns metadata (e.g. tag_list, train_start_date, train_end_date in InfluxBackedDataset class)
        """
