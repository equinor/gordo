# -*- coding: utf-8 -*-

import unittest
import tempfile

import numpy as np

from keras.wrappers.scikit_learn import BaseWrapper

from gordo_components.model import get_model
from gordo_components.model.models import KerasModel
from gordo_components.model.base import GordoBaseModel


class KerasModelTestCase(unittest.TestCase):

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

    def test_save_load(self):
        config = {
            'type': 'KerasModel',
            'kind': 'feedforward_symetric',
            'n_features': 10,
            'enc_dim': [8, 4, 2],
            'dec_dim': [2, 4, 8],
            'enc_func': ['relu', 'relu', 'relu'],
            'dec_func': ['relu', 'relu', 'relu']
        }

        # Ensure we can poke the model the same
        model = get_model(config)

        # Have to call fit, since model production is lazy
        X = np.random.random(size=100).reshape(10, 10)
        model.fit(X, X)

        xTest = np.random.random(size=100).reshape(10, 10)
        yHat = model.predict(xTest)

        with tempfile.TemporaryDirectory() as tmp:
            model.save_to_dir(tmp)
            model_clone = KerasModel.load_from_dir(tmp)

            # Assert parameters are the same.
            self.assertEqual(model.get_params(), model_clone.get_params())

            # Assert it maintained the state by ensuring predictions are the same
            self.assertTrue(
                np.allclose(yHat.flatten(), model_clone.predict(xTest).flatten())
            )
