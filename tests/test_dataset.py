# -*- coding: utf-8 -*-

import unittest

import numpy as np

from gordo_components.dataset import get_dataset
from gordo_components.dataset.base import GordoBaseDataset
from gordo_components.dataset._datasets import RandomDataset


class DatasetTestCase(unittest.TestCase):
    def test_random_dataset_attrs(self):
        """
        Test expected attributes
        """
        config = {"type": "random"}

        dataset = get_dataset(config)

        self.assertTrue(isinstance(dataset, GordoBaseDataset))
        self.assertTrue(isinstance(dataset, RandomDataset))
        self.assertTrue(hasattr(dataset, "get_data"))
        self.assertTrue(hasattr(dataset, "get_metadata"))

        X, y = dataset.get_data()
        self.assertTrue(isinstance(X, np.ndarray))

        # y can either be None or an numpy array
        self.assertTrue(isinstance(y, np.ndarray) or y is None)

        metadata = dataset.get_metadata()
        self.assertTrue(isinstance(metadata, dict))
