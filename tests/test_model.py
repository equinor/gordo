# -*- coding: utf-8 -*-

import unittest
import tempfile
import logging

import numpy as np

from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, TimeSeriesSplit

from keras.wrappers.scikit_learn import BaseWrapper

from gordo_components.model import get_model
from gordo_components.model.models import KerasAutoEncoder
from gordo_components.model.base import GordoBase


logger = logging.getLogger(__name__)


class KerasModelTestCase(unittest.TestCase):
    def test_keras_type_config(self):
        """
        Test creating a keras based model from config
        """
        config = {
            "type": "KerasAutoEncoder",
            "kind": "feedforward_model",
            "n_features": 10,
            "enc_dim": [8, 4, 2],
            "dec_dim": [2, 4, 8],
            "enc_func": ["relu", "relu", "relu"],
            "dec_func": ["relu", "relu", "relu"],
        }

        # Ensure we can poke the model the same
        model = get_model(config)
        self.assertIsInstance(model, GordoBase)
        self.assertIsInstance(model, BaseWrapper)
        self.assertIsInstance(model, KerasAutoEncoder)

    def test_keras_autoencoder_type(self):
        config = {
            "type": "KerasAutoEncoder",
            "kind": "feedforward_model",
            "n_features": 10,
            "enc_dim": [8, 4, 2],
            "dec_dim": [2, 4, 8],
            "enc_func": ["relu", "relu", "relu"],
            "dec_func": ["relu", "relu", "relu"],
        }
        model = get_model(config)
        self.assertTrue(isinstance(model, KerasAutoEncoder))
        self.assertTrue(isinstance(model, GordoBase))

    def test_keras_autoencoder_scoring(self):
        """
        Test the KerasAutoEncoder has a working scoring function
        """
        raw_model = KerasAutoEncoder(kind="feedforward_model")
        pipe = Pipeline([("ae", KerasAutoEncoder(kind="feedforward_model"))])

        X = np.random.random(size=1000).reshape(-1, 10)

        for model in (raw_model, pipe):

            with self.assertRaises(NotFittedError):
                model.score(X.copy(), X.copy())

            model.fit(X)

            score = model.score(X)
            logger.info(f"Score: {score:.4f}")

    def test_keras_autoencoder_crossval(self):
        """
        Test ability for cross validation
        """
        raw_model = KerasAutoEncoder(kind="feedforward_model")
        pipe = Pipeline([("ae", KerasAutoEncoder(kind="feedforward_model"))])

        X = np.random.random(size=1000).reshape(-1, 10)

        for model in (raw_model, pipe):
            scores = cross_val_score(
                model, X, X, cv=TimeSeriesSplit(n_splits=5, max_train_size=100)
            )
            logger.info(
                f"Mean score: {scores.mean():.4f} - Std score: {scores.std():.4f}"
            )

    def test_save_load(self):
        config = {
            "type": "KerasBaseEstimator",
            "kind": "feedforward_model",
            "n_features": 10,
            "enc_dim": [8, 4, 2],
            "dec_dim": [2, 4, 8],
            "enc_func": ["relu", "relu", "relu"],
            "dec_func": ["relu", "relu", "relu"],
        }

        # Ensure we can poke the model the same
        model = get_model(config)

        # Have to call fit, since model production is lazy
        X = np.random.random(size=100).reshape(10, 10)

        # Unless it's the KerasAutoEncoder type, it would expect a y as a target
        with self.assertRaises(TypeError):
            model.fit(X)

        # AutoEncoder is fine without a y target
        config["type"] = "KerasAutoEncoder"
        model = get_model(config)
        model.fit(X)

        xTest = np.random.random(size=100).reshape(10, 10)
        xHat = model.transform(xTest)

        with tempfile.TemporaryDirectory() as tmp:
            model.save_to_dir(tmp)
            model_clone = KerasAutoEncoder.load_from_dir(tmp)

            # Assert parameters are the same.
            self.assertEqual(model.get_params(), model_clone.get_params())

            # Assert it maintained the state by ensuring predictions are the same
            self.assertTrue(
                np.allclose(xHat.flatten(), model_clone.transform(xTest).flatten())
            )
