# -*- coding: utf-8 -*-

import unittest
import logging
import json
import os

from tempfile import TemporaryDirectory

import pytest
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import TransformedTargetRegressor
from gordo.machine.model.anomaly.diff import DiffBasedAnomalyDetector

from gordo.machine.model.models import KerasAutoEncoder
from gordo import serializer


logger = logging.getLogger(__name__)


class PipelineSerializationTestCase(unittest.TestCase):
    def test_pipeline_serialization(self):

        pipe = Pipeline(
            [
                ("pca1", PCA(n_components=10)),
                (
                    "fu",
                    FeatureUnion(
                        [
                            ("pca2", PCA(n_components=3)),
                            (
                                "pipe",
                                Pipeline(
                                    [
                                        ("minmax", MinMaxScaler()),
                                        ("truncsvd", TruncatedSVD(n_components=7)),
                                    ]
                                ),
                            ),
                        ]
                    ),
                ),
                ("ae", KerasAutoEncoder(kind="feedforward_hourglass")),
            ]
        )

        X = np.random.random(size=100).reshape(10, 10)
        pipe.fit(X.copy(), X.copy())

        with TemporaryDirectory() as tmp:

            # Test dump
            metadata = {"key": "value"}
            serializer.dump(pipe, tmp, metadata=metadata)

            # Test load from the serialized pipeline above
            pipe_clone = serializer.load(tmp)
            metadata_clone = serializer.load_metadata(tmp)

            # Ensure the metadata was saved and loaded back
            self.assertEqual(metadata, metadata_clone)

            # Verify same state for both pipelines
            y_hat_pipe1 = pipe.predict(X.copy()).flatten()
            y_hat_pipe2 = pipe_clone.predict(X.copy()).flatten()
            self.assertTrue(np.allclose(y_hat_pipe1, y_hat_pipe2))

            # Now use dumps/loads
            serialized = serializer.dumps(pipe)
            pipe_clone = serializer.loads(serialized)

            # Verify same state for both pipelines
            y_hat_pipe1 = pipe.predict(X.copy()).flatten()
            y_hat_pipe2 = pipe_clone.predict(X.copy()).flatten()
            self.assertTrue(np.allclose(y_hat_pipe1, y_hat_pipe2))


@pytest.mark.parametrize(
    "model",
    [
        KerasAutoEncoder(kind="feedforward_hourglass"),
        DiffBasedAnomalyDetector(
            base_estimator=TransformedTargetRegressor(
                regressor=KerasAutoEncoder(kind="feedforward_symmetric"),
                transformer=MinMaxScaler(),
            )
        ),
        TransformedTargetRegressor(
            regressor=Pipeline(
                steps=[
                    ("stp1", MinMaxScaler()),
                    ("stp2", KerasAutoEncoder(kind="feedforward_symmetric")),
                ]
            )
        ),
    ],
)
def test_dump_load_models(model):

    X = np.random.random(size=100).reshape(10, 10)
    model.fit(X.copy(), X.copy())
    model_out = model.predict(X.copy())

    with TemporaryDirectory() as tmp:
        serializer.dump(model, tmp)

        model_clone = serializer.load(tmp)
        model_clone_out = model_clone.predict(X.copy())

        assert np.allclose(model_out.flatten(), model_clone_out.flatten())


@pytest.mark.parametrize("location", ("metadata.json", "../metadata.json", None))
def test_load_metadata(tmpdir, location):
    """
    Test load_metadata can look in directory given as well as directory above that
    along with dealing with 'FileNotFoundError' when a non-existent file is given.
    """
    model_dir = os.path.join(tmpdir, "some-model-dir")
    os.mkdir(model_dir)
    if location:
        with open(os.path.join(model_dir, location), "w") as f:
            json.dump(dict(key="value"), f)
        assert serializer.load_metadata(model_dir) == dict(key="value")
    else:
        # Attempting to load a file which doesn't exist will raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            assert serializer.load_metadata(tmpdir)
