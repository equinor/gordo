# -*- coding: utf-8 -*-

import pytest
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

from gordo_components.server import model_io


@pytest.mark.parametrize(
    "model",
    [
        RandomForestClassifier(n_estimators=10),
        Pipeline([("clf", RandomForestClassifier(n_estimators=10))]),
        Pipeline(
            [("mm", MinMaxScaler()), ("clf", RandomForestClassifier(n_estimators=10))]
        ),
        Pipeline([("pca", PCA())]),
    ],
)
def test_model_mixin_get_model_output(model):
    """
    Test ModelMixin which ML Server uses to get model outputs. Basically every model, whether in a Pipeline
    or not, should give back a numpy array; even if the last model doesn't implement predict and just transform.
    """
    X, y = np.random.random((10, 10)), np.random.randint(low=0, high=4, size=10)
    model.fit(X, y)
    out = model_io.get_model_output(model, X)
    assert isinstance(out, np.ndarray)


def test_model_mixin_get_transformed_input():
    """
    Using ModelMixin, ensure we can get the get the transformed data which went
    to the final model in the pipeline and if it's not a pipeline, should return the original data.
    """
    X, y = np.random.random((10, 10)), np.random.randint(low=0, high=4, size=10)

    # Single model not wrapped in a Pipeline should have a 'transformed-input' equal to the original
    model = RandomForestClassifier(n_estimators=10)
    model.fit(X, y)
    out = model_io.get_transformed_input(model, X)
    assert np.allclose(X, out)

    # Pipeline with a single step which doesn't implement transform,
    # should also have 'transformed-input' equal to origianl
    model = Pipeline([("clf", RandomForestClassifier(n_estimators=10))])
    model.fit(X, y)
    out = model_io.get_transformed_input(model, X)
    assert np.allclose(X, out)

    # Pipeline with a step that does a proper transform should have an output which matches that transformer's out
    model = Pipeline(
        [("mm", MinMaxScaler()), ("clf", RandomForestClassifier(n_estimators=10))]
    )
    model.fit(X, y)
    out = model_io.get_transformed_input(model, X)
    expected = model.steps[0][1].transform(X)
    assert np.allclose(out, expected)

    # If it's a pipeline which has a transformer as it's only step, it should get the original input as well.
    model = Pipeline([("pca", PCA())])
    model.fit(X, y)
    out = model_io.get_transformed_input(model, X)
    assert np.allclose(out, X)

    # And if it's a multi-step transformer-only pipeline, it should get the output of the second to last transformer
    model = Pipeline([("mm", MinMaxScaler()), ("pca", PCA())])
    model.fit(X, y)
    out = model_io.get_transformed_input(model, X)
    expected = model.steps[0][1].transform(X)
    assert np.allclose(out, expected)


def test_model_mixin_get_inverse_transformed():
    """
    Test getting the inverse transformed of the model pipeline
    """
    X, y = np.random.random((10, 10)), np.random.randint(low=0, high=4, size=10)

    # Single model not wrapped in a Pipeline should get the original
    model = RandomForestClassifier(n_estimators=10)
    model.fit(X, y)
    out = model_io.get_inverse_transformed_input(model, X)
    assert np.allclose(X, out)

    # Pipeline with a step does have a inverse capable transformer should give back original
    model = Pipeline([("mm", MinMaxScaler()), ("pca", PCA())])
    model.fit(X, y)
    transformed_x = model.transform(X)
    out = model_io.get_inverse_transformed_input(model, transformed_x)
    assert np.allclose(X, out)

    # If it's a pipeline which has a transformer as it's only step, it should get the original input as well.
    model = Pipeline([("pca", PCA())])
    model.fit(X, y)
    transformed_x = model.transform(X)
    out = model_io.get_inverse_transformed_input(model, transformed_x)
    assert np.allclose(X, out)
