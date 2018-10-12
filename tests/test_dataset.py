# -*- coding: utf-8 -*-

import unittest
from gordo_components.dataset import Dataset


class DatasetTestCase(unittest.TestCase):

    def test_attrs(self):
        """
        Test expected attributes
        """
        self.assertTrue(hasattr(Dataset, 'get_train'))
        self.assertTrue(hasattr(Dataset, 'get_test'))
