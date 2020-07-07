# -*- coding: utf-8 -*-

import pickle
import logging
import pydoc

import pytest
import numpy as np

from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, TimeSeriesSplit

from tensorflow.keras.wrappers.scikit_learn import KerasRegressor as BaseWrapper
from tensorflow.keras.callbacks import EarlyStopping

from tests.utils import get_model
from gordo.machine.model.models import (
    KerasLSTMAutoEncoder,
    KerasLSTMForecast,
    KerasLSTMBaseEstimator,
    KerasBaseEstimator,
    KerasAutoEncoder,
    create_keras_timeseriesgenerator,
)
from gordo.machine.model.factories import lstm_autoencoder
from gordo.machine.model.base import GordoBase
from gordo.machine.model.register import register_model_builder


logger = logging.getLogger(__name__)


# Generator of model types to each available model kind for that type
# ie. (('KerasAutoEncoder', 'hourglass'), ('KerasAutoEncoder', 'symmetric'), ...)
MODEL_COMBINATIONS = list(
    (model, kind)
    for model in register_model_builder.factories.keys()
    for kind in register_model_builder.factories[model].keys()
)

# Generator of model types to one kind, rather than all combinations of kinds per type
MODEL_SINGLE_KIND = list(
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


@pytest.mark.parametrize("n_features_out", (2, 3))
@pytest.mark.parametrize("model,kind", MODEL_SINGLE_KIND)
def test_keras_autoencoder_scoring(model, kind, n_features_out):
    """
    Test the KerasAutoEncoder and KerasLSTMAutoEncoder have a working scoring function
    """
    Model = pydoc.locate(f"gordo.machine.model.models.{model}")
    model = Pipeline([("model", Model(kind=kind))])
    X = np.random.random((8, 2))

    # Should be able to deal with y output different than X input features
    y = np.random.random((8, n_features_out))

    with pytest.raises(NotFittedError):
        model.score(X, y)

    model.fit(X, y)
    score = model.score(X, y)
    logger.info(f"Score: {score:.4f}")


@pytest.mark.parametrize("model,kind", MODEL_SINGLE_KIND)
def test_keras_autoencoder_crossval(model, kind):
    """
    Test ability for cross validation
    """
    Model = pydoc.locate(f"gordo.machine.model.models.{model}")
    model = Pipeline([("model", Model(kind=kind))])

    X = np.random.random(size=(15, 2))
    y = X.copy()

    scores = cross_val_score(
        model, X, y, cv=TimeSeriesSplit(n_splits=2, max_train_size=2)
    )
    assert isinstance(scores, np.ndarray)
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
    assert isinstance(model_out, pydoc.locate(f"gordo.machine.model.models.{model}"))


@pytest.mark.parametrize("model,kind", MODEL_SINGLE_KIND)
def test_save_load(model, kind):
    config = {"type": model, "kind": kind}

    # Have to call fit, since model production is lazy
    X = np.random.random(size=10).reshape(5, 2)
    y = X.copy()

    # AutoEncoder is fine without a y target
    config["type"] = model
    model_out = get_model(config)
    if model == KerasLSTMForecast:
        assert "forecast_steps" in model_out.get_metadata()

    assert "history" not in model_out.get_metadata()
    model_out.fit(X, y)
    assert "history" in model_out.get_metadata()

    xTest = np.random.random(size=6).reshape(3, 2)
    xHat = model_out.predict(xTest)

    model_out_clone = pickle.loads(pickle.dumps(model_out))

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


def test_lookback_window_ae_valueerror_during_fit():
    """
    Assert (for LSTMAutoEncoder) ValueError
    is raised in fit method if lookback_window > number of readings (rows of X)
    """
    lookback_window = 11
    with pytest.raises(ValueError):
        model = KerasLSTMAutoEncoder(
            kind=lstm_autoencoder.lstm_model, lookback_window=lookback_window
        )
        X, y = np.random.rand(10), np.random.rand(10)
        model.fit(X, y)


def test_keras_ae_reshapes_array():
    """
    Asserts KerasLSTMAutoEncoder accepts an array of elements, which it will
    reshape into the matrix of single elements it needs
    """
    model = KerasLSTMAutoEncoder(kind=lstm_autoencoder.lstm_model)
    X, y = np.random.rand(100), np.random.rand(100)
    model.fit(X, y)
    model.predict(X)


def test_keras_forecast_reshapes_array():
    """
    Asserts KerasLSTMForecast accepts an array of elements, which it will
    reshape into the matrix of single elements it needs
    """
    model = KerasLSTMForecast(kind=lstm_autoencoder.lstm_model)
    X, y = np.random.rand(100), np.random.rand(100)
    model.fit(X, y)
    model.predict(X)


def test_lookback_window_ae_valueerror_during_predict():
    """
    Assert (for LSTMAutoEncoder) ValueError
    is raised in fit method if lookback_window > number of readings (rows of X)
    """
    model = KerasLSTMAutoEncoder(kind=lstm_autoencoder.lstm_model, lookback_window=3)
    xTrain, yTrain = np.random.random(size=(4, 2)), np.random.random(size=(4, 2))
    model.fit(xTrain, yTrain)
    with pytest.raises(ValueError):
        model.predict(xTrain[-3:-1, :])


@pytest.mark.parametrize("lookback_window", (5, 6))
def test_lookback_window_forecast_valueerror_during_fit(lookback_window):
    """
    Assert (for LSTMForecast) ValueError is raised
    in fit method if lookback_window >= number of readings (rows of X)
    """
    model = KerasLSTMForecast(
        kind=lstm_autoencoder.lstm_model, lookback_window=lookback_window
    )
    with pytest.raises(ValueError):
        X = np.random.random(size=(5, 2))
        y = X.copy()
        model.fit(X, y)


@pytest.mark.parametrize("lookback_window", (5, 6))
def test_lookback_window_forecast_valueerror_during_predict(lookback_window: int):
    """
    Assert (for LSTMForecast) ValueError is raised
    in fit method if lookback_window >= number of readings (rows of X)
    """
    X = np.random.random(size=(5, 2))
    y = X.copy()
    model = KerasLSTMForecast(
        kind=lstm_autoencoder.lstm_model, lookback_window=lookback_window
    )
    with pytest.raises(ValueError):
        model.fit(X, y)


def test_create_keras_timeseriesgenerator_lb3_loah0_bs2():
    """Check that right output is generated from create_keras_timeseriesgenerator"""
    X = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
    y = X.copy()
    gen = create_keras_timeseriesgenerator(
        X, y, batch_size=2, lookback_window=3, lookahead=0
    )
    batch_1 = gen[0]
    batch_2 = gen[1]

    batch_1_x = batch_1[0].tolist()
    batch_1_y = batch_1[1].tolist()

    batch_2_x = batch_2[0].tolist()
    batch_2_y = batch_2[1].tolist()

    assert [[[0, 1], [2, 3], [4, 5]], [[2, 3], [4, 5], [6, 7]]] == batch_1_x
    assert [[4, 5], [6, 7]] == batch_1_y

    assert [[[4, 5], [6, 7], [8, 9]]] == batch_2_x
    assert [[8, 9]] == batch_2_y


def test_create_keras_timeseriesgenerator_lb2_loah1_bs2():
    """
    Check right output is generated from create_keras_timeseriesgenerator
    We use lookback_window 2 to get some more interesting batches with lookahead 1
    """
    X = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
    y = X.copy()
    gen = create_keras_timeseriesgenerator(
        X, y, batch_size=2, lookback_window=2, lookahead=1
    )
    batch_1 = gen[0]
    batch_2 = gen[1]

    batch_1_x = batch_1[0].tolist()
    batch_1_y = batch_1[1].tolist()

    batch_2_x = batch_2[0].tolist()
    batch_2_y = batch_2[1].tolist()

    assert [[[0, 1], [2, 3]], [[2, 3], [4, 5]]] == batch_1_x
    assert [[4, 5], [6, 7]] == batch_1_y

    assert [[[4, 5], [6, 7]]] == batch_2_x
    assert [[8, 9]] == batch_2_y


def test_create_keras_timeseriesgenerator_lb3_loah2_bs2():
    """
    Check right output is generated from create_keras_timeseriesgenerator
    We use lookback_window 2 to get some more interesting batches with lookahead 1
    """
    X = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
    y = X.copy()
    gen = create_keras_timeseriesgenerator(
        X, y, batch_size=2, lookback_window=2, lookahead=2
    )
    batch_1 = gen[0]
    batch_2 = gen[1]

    batch_1_x = batch_1[0].tolist()
    batch_1_y = batch_1[1].tolist()

    batch_2_x = batch_2[0].tolist()
    batch_2_y = batch_2[1].tolist()

    assert [[[0, 1], [2, 3]], [[2, 3], [4, 5]]] == batch_1_x
    assert [[6, 7], [8, 9]] == batch_1_y

    assert [] == batch_2_x  # No more elements left
    assert [] == batch_2_y


def test_create_keras_timeseriesgenerator_raise_error_on_neg_lookahead():
    """Check create_keras_timeseriesgenerator raises an error on negative lookahead"""
    X = np.array([[0, 1]])
    y = X.copy()
    with pytest.raises(ValueError):
        create_keras_timeseriesgenerator(
            X, y, batch_size=2, lookback_window=2, lookahead=-1
        )


def test_lstmae_predict_output():
    """
    test for KerasLSTMAutoEncoder
      - test dimension of output
      - test that first half of output is testing data
    """
    xTrain, yTrain = np.random.random(size=(5, 3)), np.random.random((5, 3))
    lookback_window = 3
    model = KerasLSTMAutoEncoder(
        kind=lstm_autoencoder.lstm_model, lookback_window=lookback_window
    )
    model = model.fit(xTrain, yTrain)
    xTest = np.random.random(size=(4, 3))
    out = model.predict(xTest)
    assert out.shape == (2, 3)


def test_keras_autoencoder_fits_callbacks():
    model = KerasAutoEncoder(
        kind="feedforward_hourglass",
        batch_size=128,
        callbacks=[
            {
                "tensorflow.keras.callbacks.EarlyStopping": {
                    "monitor": "val_loss",
                    "patience": 10,
                }
            }
        ],
    )
    sk_params = model.sk_params
    assert len(sk_params["callbacks"]) == 1
    first_callback = sk_params["callbacks"][0]
    assert isinstance(first_callback, EarlyStopping)
    assert first_callback.monitor == "val_loss"
    assert first_callback.patience == 10
