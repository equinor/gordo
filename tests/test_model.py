# -*- coding: utf-8 -*-

import unittest

from gordo_components.model import get_model
from gordo_components.model._models import GordoKerasModel
from gordo_components.model.base import GordoBaseModel


class ModelTestCase(unittest.TestCase):

    def test_keras_type_config(self):
        """
        Test creating a keras based model from config
        """
        config = {
            'type'      : 'keras',
            'n_features': 10,
            'enc_dim'   : [8, 4, 2],
            'dec_dim'   : [2, 4, 8],
            'enc_func'  : ['relu', 'relu', 'relu'],
            'dec_func'  : ['relu', 'relu', 'relu']
        }

        # Should produce a model inherited from GordoBaseModel, 
        # so we can poke it the same
        model = get_model(config)
        self.assertIsInstance(model, GordoBaseModel)

        # From keras config should produce a keras model
        from keras.models import Model as KerasModel
        self.assertIsInstance(model.model, KerasModel)
 