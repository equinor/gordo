# -*- coding: utf-8 -*-

import unittest

from gordo_components.model.factories.model_factories_utils import hourglass_calc_dims


class ModelFactoriesUtilsTestCase(unittest.TestCase):
    def test_hourglass_calc_dims_check_dims(self):
        """
        Test that hourglass_calc_dims implements the correct dimensions
        """

        dims = hourglass_calc_dims(0.2, 4, 5)
        self.assertEqual(dims, [4, 3, 2, 1])
        dims = hourglass_calc_dims(0.5, 3, 10)
        self.assertEqual(dims, [8, 7, 5])
        dims = hourglass_calc_dims(0.5, 3, 3)
        self.assertEqual(dims, [3, 2, 2])
        dims = hourglass_calc_dims(0.3, 3, 10)
        self.assertEqual(dims, [8, 5, 3])
        dims = hourglass_calc_dims(1, 3, 10)
        self.assertEqual(dims, [10, 10, 10])
        dims = hourglass_calc_dims(0, 3, 100000)
        self.assertEqual(dims, [66667, 33334, 1])
