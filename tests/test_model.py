# -*- coding: utf-8 -*-

import unittest
import tempfile
import logging
import pydoc

import numpy as np

from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, TimeSeriesSplit

from keras.wrappers.scikit_learn import BaseWrapper

from gordo_components.model import get_model
from gordo_components.model.models import KerasLSTMAutoEncoder, KerasLSTMForecast
from gordo_components.model.factories import lstm_autoencoder
from gordo_components.model.base import GordoBase
from gordo_components.model.register import register_model_builder


logger = logging.getLogger(__name__)


class KerasModelTestCase(unittest.TestCase):
    def test_keras_type_config(self):
        """
        Test creating a keras based model from config
        """
        self.factories = register_model_builder.factories
        for model in self.factories.keys():
            if model not in ["KerasBaseEstimator", "KerasLSTMBaseEstimator"]:
                for model_kind in self.factories[model].keys():
                    config = {"type": model, "kind": model_kind}

                    # Ensure we can poke the model the same
                    model_out = get_model(config)
                    self.assertIsInstance(model_out, GordoBase)
                    self.assertIsInstance(model_out, BaseWrapper)
                    self.assertIsInstance(
                        model_out,
                        pydoc.locate(f"gordo_components.model.models.{model}"),
                    )

    def test_keras_autoencoder_forecast_scoring(self):
        """
        Test the KerasAutoEncoder, KerasLSTMAutoEncoder and KerasLSTMForecast have a working scoring function
        """
        self.factories = register_model_builder.factories
        X = np.random.random(size=8).reshape(-1, 2)
        for model_str in self.factories.keys():
            if model_str not in ["KerasBaseEstimator", "KerasLSTMBaseEstimator"]:
                for model_kind in self.factories[model_str].keys():
                    Model = pydoc.locate(f"gordo_components.model.models.{model_str}")
                    raw_model = Model(kind=model_kind)
                    pipe = Pipeline([("ae", Model(kind=model_kind))])

                    for model in (raw_model, pipe):

                        with self.assertRaises(NotFittedError):
                            model.score(X.copy(), X.copy())

                        model.fit(X)
                        score = model.score(X)
                        logger.info(f"Score: {score:.4f}")

    def test_keras_autoencoder_forecast_crossval(self):
        """
        Test ability for cross validation
        """
        self.factories = register_model_builder.factories
        X = np.random.random(size=(15, 2))
        for model_str in self.factories.keys():
            if model_str not in ["KerasBaseEstimator", "KerasLSTMBaseEstimator"]:
                for model_kind in self.factories[model_str].keys():
                    Model = pydoc.locate(f"gordo_components.model.models.{model_str}")
                    raw_model = Model(kind=model_kind)
                    pipe = Pipeline([("ae", Model(kind=model_kind))])
                    for model in (raw_model, pipe):
                        scores = cross_val_score(
                            model,
                            X,
                            X,
                            cv=TimeSeriesSplit(n_splits=2, max_train_size=2),
                        )
                        logger.info(
                            f"Mean score: {scores.mean():.4f} - Std score: {scores.std():.4f}"
                        )

    def test_expected_target_in_fit(
        self
    ):  # TODO to remove if we unregister feedfoward_autoencoder from KerasBaseEstimator?
        config = {"type": "KerasBaseEstimator", "kind": "feedforward_hourglass"}

        # Ensure we can poke the model the same
        model = get_model(config)

        # Have to call fit, since model production is lazy
        X = np.random.random(size=10).reshape(5, 2)

        # Unless it's the KerasAutoEncoder type, it would expect a y as a target
        with self.assertRaises(TypeError):
            model.fit(X)

    def test_save_load(self):
        self.factories = register_model_builder.factories
        X = np.random.random(size=10).reshape(5, 2)
        for model in self.factories.keys():
            if model not in ["KerasBaseEstimator", "KerasLSTMBaseEstimator"]:
                for model_kind in self.factories[model].keys():
                    config = {"type": model, "kind": model_kind}

                    # Have to call fit, since model production is lazy
                    # AutoEncoder is fine without a y target
                    config["type"] = model
                    model_out = get_model(config)
                    self.assertFalse("history" in model_out.get_metadata())
                    model_out.fit(X)
                    self.assertTrue("history" in model_out.get_metadata())

                    xTest = np.random.random(size=6).reshape(3, 2)
                    xHat = model_out.transform(xTest)

                    with tempfile.TemporaryDirectory() as tmp:
                        model_out.save_to_dir(tmp)
                        model_out_clone = pydoc.locate(
                            f"gordo_components.model.models.{model}"
                        ).load_from_dir(tmp)

                        # Assert parameters are the same.
                        self.assertEqual(
                            model_out_clone.get_params(), model_out_clone.get_params()
                        )

                        # Assert it maintained the state by ensuring predictions are the same
                        self.assertTrue(
                            np.allclose(
                                xHat.flatten(),
                                model_out_clone.transform(xTest).flatten(),
                            )
                        )

                        self.assertTrue("history" in model_out.get_metadata())
                        self.assertEqual(
                            model_out.get_metadata(),
                            model_out_clone.get_metadata(),
                            "Metadata from model is not same after "
                            "saving and loading",
                        )
                        # Assert that epochs list, history dict and params dict in
                        # the History object are the same
                        self.assertEqual(
                            model_out.model.history.epoch,
                            model_out_clone.model.history.epoch,
                            "Epoch lists differ between original and "
                            "loaded model history",
                        )
                        self.assertEqual(
                            model_out.model.history.history,
                            model_out_clone.model.history.history,
                            "History dictionary with losses and "
                            "accuracies differ "
                            "between original and loaded model history",
                        )
                        self.assertEqual(
                            model_out.model.history.params,
                            model_out_clone.model.history.params,
                            "Params dictionaries differ between "
                            "original and loaded "
                            "model history",
                        )

    def test__generate_window_lookback_window_ae_valueerror(self):

        # Assert that (for LSTMAutoEncoder) ValueError
        # is raised if lookback_window > number of readings (rows of X)

        X = np.random.random(size=20).reshape(10, 2)
        lookback_window = 11
        with self.assertRaises(ValueError):
            model = KerasLSTMAutoEncoder(
                kind=lstm_autoencoder.lstm_model, lookback_window=lookback_window
            )
            gen = model.generate_window(X, architecture="lstm_autoencoder")
            next(gen)

    def test_generate_window_lookback_window_forecast_valueerror(self):
        # Assert that (for LSTMForecast) ValueError is raised if lookback_window >= number of readings (rows of X)
        X = np.random.random(size=20).reshape(10, 2)
        for lookback_window in [10, 11]:
            with self.assertRaises(ValueError):
                model = KerasLSTMForecast(
                    kind=lstm_autoencoder.lstm_model, lookback_window=lookback_window
                )
                gen = model.generate_window(X, architecture="lstm_forecast")
                next(gen)

    def test_generate_window_architecture_valueerror(self):
        # Assert that ValueError is raised if architecture not 'lstm_autoencoder' or 'lstm_forecast'

        X = np.random.random(size=(5, 2))
        lookback_window = 5
        with self.assertRaises(ValueError):
            model = KerasLSTMForecast(
                kind=lstm_autoencoder.lstm_model, lookback_window=lookback_window
            )
            gen = model.generate_window(X, architecture="forecast")
            next(gen)
            model = KerasLSTMAutoEncoder(
                kind=lstm_autoencoder.lstm_model, lookback_window=lookback_window
            )
            gen = model.generate_window(X, architecture="autoencoder")
            next(gen)

    def test_generate_window_lstm_ae_output(self):
        # Check that right output is generated from generate_window (architecture = 'lstm_autoencoder')
        X = np.random.random(size=10).reshape(5, 2)

        lookback_window = 4
        model = KerasLSTMAutoEncoder(
            kind=lstm_autoencoder.lstm_model, lookback_window=lookback_window
        )
        gen = model.generate_window(X, architecture="lstm_autoencoder")
        gen_out_1 = next(gen)
        gen_out_2 = next(gen)
        self.assertEqual(gen_out_1[0].tolist(), X[0:4].reshape(1, 4, 2).tolist())
        self.assertEqual(gen_out_2[0].tolist(), X[1:5].reshape(1, 4, 2).tolist())
        self.assertEqual(gen_out_1[1].tolist(), X[3].reshape(1, 2).tolist())
        self.assertEqual(gen_out_2[1].tolist(), X[4].reshape(1, 2).tolist())

        X = np.random.random(size=8).reshape(4, 2)
        lookback_window = 2
        model = KerasLSTMAutoEncoder(
            kind=lstm_autoencoder.lstm_model, lookback_window=lookback_window
        )
        gen_no_y = model.generate_window(
            X, architecture="lstm_autoencoder", output_y=False
        )
        gen_no_y_out_1 = next(gen_no_y)
        gen_no_y_out_2 = next(gen_no_y)
        gen_no_y_out_3 = next(gen_no_y)
        self.assertEqual(gen_no_y_out_1.tolist(), X[0:2].reshape(1, 2, 2).tolist())
        self.assertEqual(gen_no_y_out_2.tolist(), X[1:3].reshape(1, 2, 2).tolist())
        self.assertEqual(gen_no_y_out_3.tolist(), X[2:4].reshape(1, 2, 2).tolist())

    def test_generate_window_lstm_forecast_output(self):
        # Check that right output is generated from generate_window (architecture = 'lstm_forecast')
        X = np.random.random(size=(5, 2))

        lookback_window = 3
        model = KerasLSTMForecast(
            kind=lstm_autoencoder.lstm_model, lookback_window=lookback_window
        )
        gen = model.generate_window(X, architecture="lstm_forecast")
        gen_out_1 = next(gen)
        gen_out_2 = next(gen)
        self.assertEqual(gen_out_1[0].tolist(), X[0:3].reshape(1, 3, 2).tolist())
        self.assertEqual(gen_out_2[0].tolist(), X[1:4].reshape(1, 3, 2).tolist())
        self.assertEqual(gen_out_1[1].tolist(), X[3].reshape(1, 2).tolist())
        self.assertEqual(gen_out_2[1].tolist(), X[4].reshape(1, 2).tolist())

        X = np.random.random(size=(4, 2))
        lookback_window = 2
        model = KerasLSTMForecast(
            kind=lstm_autoencoder.lstm_model, lookback_window=lookback_window
        )
        gen_no_y = model.generate_window(
            X, architecture="lstm_forecast", output_y=False
        )
        gen_no_y_out_1 = next(gen_no_y)
        gen_no_y_out_2 = next(gen_no_y)
        self.assertEqual(gen_no_y_out_1.tolist(), X[0:2].reshape(1, 2, 2).tolist())
        self.assertEqual(gen_no_y_out_2.tolist(), X[1:3].reshape(1, 2, 2).tolist())

    def test_lstmae_transform_output(self):
        # test for KerasLSTMAutoEncoder
        #  - test dimension of output
        #  - test that first half of output is testing data

        X_train = np.random.random(size=(5, 3))
        lookback_window = 3
        model = KerasLSTMAutoEncoder(
            kind=lstm_autoencoder.lstm_model, lookback_window=lookback_window
        )
        model = model.fit(X_train)
        X_test = np.random.random(size=(4, 3))
        out = model.transform(X_test)
        self.assertEqual(out.shape, (2, 6))
        self.assertEqual(out[:, :3].tolist(), X_test[lookback_window - 1 :, :].tolist())

    def test_lstmforecast_transform_output(self):
        # test for KerasLSTMForecast
        #  - test dimension of output
        #  - test that first half of output is testing data

        X_train = np.random.random(size=(5, 3))
        lookback_window = 3
        model = KerasLSTMForecast(
            kind=lstm_autoencoder.lstm_model, lookback_window=lookback_window
        )
        model = model.fit(X_train)
        X_test = np.random.random(size=(6, 3))
        out = model.transform(X_test)
        self.assertEqual(out.shape, (3, 6))
        self.assertEqual(out[:, :3].tolist(), X_test[lookback_window:, :].tolist())
