# -*- coding: utf-8 -*-

import pytest
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

from gordo.server import model_io


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
