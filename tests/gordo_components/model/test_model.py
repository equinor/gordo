# -*- coding: utf-8 -*-

import unittest
import tempfile
import logging
import pydoc

import pytest
import numpy as np

from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, TimeSeriesSplit

from keras.wrappers.scikit_learn import BaseWrapper

from gordo_components.model import get_model
from gordo_components.model.models import (
    KerasLSTMAutoEncoder,
    KerasLSTMForecast,
    KerasLSTMBaseEstimator,
    KerasBaseEstimator,
    create_keras_timeseriesgenerator,
)
from gordo_components.model.factories import lstm_autoencoder
from gordo_components.model.base import GordoBase
from gordo_components.model.register import register_model_builder


logger = logging.getLogger(__name__)


# Generator of model types to each available model kind for that type
# ie. (('KerasAutoEncoder', 'hourglass'), ('KerasAutoEncoder', 'symmetric'), ...)
MODEL_COMBINATIONS = (
    (model, kind)
    for model in register_model_builder.factories.keys()
    for kind in register_model_builder.factories[model].keys()
)

# Generator of model types to one kind, rather than all combinations of kinds per type
MODEL_SINGLE_KIND = (
    (model, sorted(register_model_builder.factories[model].keys())[0])
    for model in register_model_builder.factories.keys()
)


@pytest.mark.parametrize("BaseModel", [KerasLSTMBaseEstimator, KerasBaseEstimator])
def test_base_class_models(BaseModel):
    """
    Test that the ABC cannot be instantiated, in that they require some implementation
    """
    with pytest.raises(TypeError):
        BaseModel()


@pytest.mark.parametrize("model,kind", MODEL_SINGLE_KIND)
def test_keras_autoencoder_scoring(model, kind):
    """
    Test the KerasAutoEncoder and KerasLSTMAutoEncoder have a working scoring function
    """
    Model = pydoc.locate(f"gordo_components.model.models.{model}")
    model = Pipeline([("model", Model(kind=kind))])
    X = np.random.random(size=8).reshape(-1, 2)

    with pytest.raises(NotFittedError):
        model.score(X.copy(), X.copy())

    model.fit(X)
    score = model.score(X, X)
    logger.info(f"Score: {score:.4f}")


@pytest.mark.parametrize("model,kind", MODEL_SINGLE_KIND)
def test_keras_autoencoder_crossval(model, kind):
    """
    Test ability for cross validation
    """
    Model = pydoc.locate(f"gordo_components.model.models.{model}")
    model = Pipeline([("model", Model(kind=kind))])

    X = np.random.random(size=(15, 2))

    scores = cross_val_score(
        model, X, X, cv=TimeSeriesSplit(n_splits=2, max_train_size=2)
    )
    logger.info(f"Mean score: {scores.mean():.4f} - Std score: {scores.std():.4f}")


@pytest.mark.parametrize("model,kind", MODEL_COMBINATIONS)
def test_keras_type_config(model, kind):
    """
    Test creating a keras based model from config
    """
    config = {"type": model, "kind": kind}

    # Ensure we can poke the model the same
    model_out = get_model(config)
    assert isinstance(model_out, GordoBase)
    assert isinstance(model_out, BaseWrapper)
    assert isinstance(model_out, pydoc.locate(f"gordo_components.model.models.{model}"))


@pytest.mark.parametrize("model,kind", MODEL_SINGLE_KIND)
def test_save_load(model, kind):
    config = {"type": model, "kind": kind}

    # Have to call fit, since model production is lazy
    X = np.random.random(size=10).reshape(5, 2)

    # AutoEncoder is fine without a y target
    config["type"] = model
    model_out = get_model(config)
    if model == KerasLSTMForecast:
        assert "forecast_steps" in model_out.get_metadata()

    assert "history" not in model_out.get_metadata()
    model_out.fit(X)
    assert "history" in model_out.get_metadata()

    xTest = np.random.random(size=6).reshape(3, 2)
    xHat = model_out.predict(xTest)

    with tempfile.TemporaryDirectory() as tmp:
        model_out.save_to_dir(tmp)
        model_out_clone = pydoc.locate(
            f"gordo_components.model.models.{model}"
        ).load_from_dir(tmp)

        # Assert parameters are the same.
        assert model_out_clone.get_params() == model_out_clone.get_params()

        # Assert it maintained the state by ensuring predictions are the same
        assert np.allclose(xHat.flatten(), model_out_clone.predict(xTest).flatten())

        assert "history" in model_out.get_metadata()
        assert (
            model_out.get_metadata() == model_out_clone.get_metadata()
        ), "Metadata from model is not same after saving and loading"

        # Assert that epochs list, history dict and params dict in
        # the History object are the same
        assert (
            model_out.model.history.epoch == model_out_clone.model.history.epoch
        ), "Epoch lists differ between original and loaded model history"

        assert (
            model_out.model.history.history == model_out_clone.model.history.history
        ), "History dictionary with losses and accuracies differ between original and loaded model history"

        assert (
            model_out.model.history.params == model_out_clone.model.history.params
        ), "Params dictionaries differ between original and loaded model history"


class KerasModelTestCase(unittest.TestCase):
    def test_lookback_window_ae_valueerror_during_fit(self):

        # Assert that (for LSTMAutoEncoder) ValueError
        # is raised in fit method if lookback_window > number of readings (rows of X)

        lookback_window = 11
        with self.assertRaises(ValueError):
            model = KerasLSTMAutoEncoder(
                kind=lstm_autoencoder.lstm_model, lookback_window=lookback_window
            )
            X = np.random.rand(10)
            model.fit(X)

    def test_keras_ae_reshapes_array(self):
        # Asserts that KerasLSTMAutoEncoder accepts an array of elements, which it will
        # reshape into the matrix of single elements it needs
        model = KerasLSTMAutoEncoder(kind=lstm_autoencoder.lstm_model)
        X = np.random.rand(100)
        model.fit(X)
        X = np.random.rand(100)
        model.predict(X)

    def test_keras_forecast_reshapes_array(self):
        # Asserts that KerasLSTMForecast accepts an array of elements, which it will
        # reshape into the matrix of single elements it needs
        model = KerasLSTMForecast(kind=lstm_autoencoder.lstm_model)
        X = np.random.rand(100)
        model.fit(X)
        X = np.random.rand(100)
        model.predict(X)

    def test_lookback_window_ae_valueerror_during_predict(self):

        # Assert that (for LSTMAutoEncoder) ValueError
        # is raised in fit method if lookback_window > number of readings (rows of X)

        lookback_window = 3
        model = KerasLSTMAutoEncoder(
            kind=lstm_autoencoder.lstm_model, lookback_window=lookback_window
        )
        X_train = np.random.random(size=(4, 2))
        model.fit(X_train)
        with self.assertRaises(ValueError):
            X_test = X_train[-3:-1, :]
            model.predict(X_test)

    def test_lookback_window_forecast_valueerror_during_fit(self):
        # Assert that (for LSTMForecast) ValueError is raised
        # in fit method if lookback_window >= number of readings (rows of X)
        for lookback_window in [5, 6]:
            model = KerasLSTMForecast(
                kind=lstm_autoencoder.lstm_model, lookback_window=lookback_window
            )
            with self.assertRaises(ValueError):
                X = np.random.random(size=(5, 2))
                model.fit(X)

    def test_lookback_window_forecast_valueerror_during_predict(self):
        # Assert that (for LSTMForecast) ValueError is raised
        # in fit method if lookback_window >= number of readings (rows of X)
        X = np.random.random(size=(5, 2))
        for lookback_window in [5, 6]:
            model = KerasLSTMForecast(
                kind=lstm_autoencoder.lstm_model, lookback_window=lookback_window
            )
            with self.assertRaises(ValueError):
                model.fit(X)

    def test_create_keras_timeseriesgenerator_lb3_loah0_bs2(self):
        # Check that right output is generated from create_keras_timeseriesgenerator
        X = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
        gen = create_keras_timeseriesgenerator(
            X, batch_size=2, lookback_window=3, lookahead=0
        )
        batch_1 = gen[0]
        batch_2 = gen[1]

        batch_1_x = batch_1[0].tolist()
        batch_1_y = batch_1[1].tolist()

        batch_2_x = batch_2[0].tolist()
        batch_2_y = batch_2[1].tolist()

        self.assertEquals(
            [[[0, 1], [2, 3], [4, 5]], [[2, 3], [4, 5], [6, 7]]], batch_1_x
        )
        self.assertEqual([[4, 5], [6, 7]], batch_1_y)

        self.assertEqual([[[4, 5], [6, 7], [8, 9]]], batch_2_x)
        self.assertEqual([[8, 9]], batch_2_y)

    def test_create_keras_timeseriesgenerator_lb2_loah1_bs2(self):
        # Check that right output is generated from create_keras_timeseriesgenerator
        # We use lookback_window 2 to get some more interesting batches with lookahead 1
        X = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
        gen = create_keras_timeseriesgenerator(
            X, batch_size=2, lookback_window=2, lookahead=1
        )
        batch_1 = gen[0]
        batch_2 = gen[1]

        batch_1_x = batch_1[0].tolist()
        batch_1_y = batch_1[1].tolist()

        batch_2_x = batch_2[0].tolist()
        batch_2_y = batch_2[1].tolist()

        self.assertEquals([[[0, 1], [2, 3]], [[2, 3], [4, 5]]], batch_1_x)
        self.assertEqual([[4, 5], [6, 7]], batch_1_y)

        self.assertEqual([[[4, 5], [6, 7]]], batch_2_x)
        self.assertEqual([[8, 9]], batch_2_y)

    def test_create_keras_timeseriesgenerator_lb3_loah2_bs2(self):
        # Check that right output is generated from create_keras_timeseriesgenerator
        # We use lookback_window 2 to get some more interesting batches with lookahead 1
        X = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
        gen = create_keras_timeseriesgenerator(
            X, batch_size=2, lookback_window=2, lookahead=2
        )
        batch_1 = gen[0]
        batch_2 = gen[1]

        batch_1_x = batch_1[0].tolist()
        batch_1_y = batch_1[1].tolist()

        batch_2_x = batch_2[0].tolist()
        batch_2_y = batch_2[1].tolist()

        self.assertEquals([[[0, 1], [2, 3]], [[2, 3], [4, 5]]], batch_1_x)
        self.assertEqual([[6, 7], [8, 9]], batch_1_y)

        self.assertEqual([], batch_2_x)  # No more elements left
        self.assertEqual([], batch_2_y)

    def test_create_keras_timeseriesgenerator_raise_error_on_neg_lookahead(self):
        # Check that create_keras_timeseriesgenerator raises an error on negative
        # lookahead
        X = np.array([[0, 1]])
        with self.assertRaises(ValueError):
            create_keras_timeseriesgenerator(
                X, batch_size=2, lookback_window=2, lookahead=-1
            )

    def test_lstmae_predict_output(self):
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
        out = model.predict(X_test)
        self.assertEqual(out.shape, (2, 3))
