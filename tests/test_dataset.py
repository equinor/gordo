# -*- coding: utf-8 -*-

import unittest
from gordo_components.dataset import get_dataset
from gordo_components.dataset.base import GordoBaseDataset
from gordo_components.dataset._datasets import RandomDataset


class DatasetTestCase(unittest.TestCase):

    def test_random_dataset_attrs(self):
        """
        Test expected attributes
        """
        config = {
            'type': 'random'
        }

        dataset = get_dataset(config)

        self.assertTrue(isinstance(dataset, GordoBaseDataset))
        self.assertTrue(isinstance(dataset, RandomDataset))
        self.assertTrue(hasattr(dataset, 'get_train'))
        self.assertTrue(hasattr(dataset, 'get_test'))
