# -*- coding: utf-8 -*-

import unittest

from keras.wrappers.scikit_learn import BaseWrapper

from gordo_components.model import get_model
from gordo_components.model.models import KerasModel
from gordo_components.model.base import GordoBaseModel


class ModelTestCase(unittest.TestCase):

    def test_keras_type_config(self):
        """
        Test creating a keras based model from config
        """
        config = {
            'type'      : 'KerasModel',
            'kind'      : 'feedforward_symetric',
            'n_features': 10,
            'enc_dim'   : [8, 4, 2],
            'dec_dim'   : [2, 4, 8],
            'enc_func'  : ['relu', 'relu', 'relu'],
            'dec_func'  : ['relu', 'relu', 'relu']
        }

        # Ensure we can poke the model the same
        model = get_model(config)
        self.assertIsInstance(model, GordoBaseModel)
        self.assertIsInstance(model, BaseWrapper)
        self.assertIsInstance(model, KerasModel)
