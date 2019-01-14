# -*- coding: utf-8 -*-

import unittest
import logging

from collections import OrderedDict
from os import path, listdir
from tempfile import TemporaryDirectory

import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import MinMaxScaler

from gordo_components.model.models import KerasAutoEncoder
from gordo_components import serializer


logger = logging.getLogger(__name__)


class PipelineSerializationTestCase(unittest.TestCase):
    def _structure_verifier(self, prefix_dir, structure):
        """
        Recursively check directory / file structure as represented in an
        OrderedDict
        """
        for directory, file_or_dict in structure.items():

            # Join the prefix_dir to the relative directory for this key/value
            directory = path.join(prefix_dir, directory)

            logger.debug(f"Prefix dir listing: {listdir(prefix_dir)}")
            logger.debug(f"Dir: {directory}")
            logger.debug(f"File or dict: {file_or_dict}")
            logger.debug(f"Files in dir: {listdir(directory)}")
            logger.debug("-" * 30)

            self.assertTrue(path.isdir(directory))

            # If this is an OrderedDict, then it's another subdirectory struct
            if isinstance(file_or_dict, OrderedDict):
                self._structure_verifier(directory, file_or_dict)
            else:

                # Otherwise we just need to verify that this is indeed a file
                self.assertTrue(path.isfile(path.join(directory, file_or_dict)))

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
                ("ae", KerasAutoEncoder(kind="feedforward_symetric")),
            ]
        )

        X = np.random.random(size=100).reshape(10, 10)
        pipe.fit(X.copy(), X.copy())

        with TemporaryDirectory() as tmp:

            # Test dump
            metadata = {"key": "value"}
            serializer.dump(pipe, tmp, metadata=metadata)

            # Assert that a dirs are created for each step in Pipeline
            expected_structure = OrderedDict(
                [
                    ("n_step=000-class=sklearn.pipeline.Pipeline", "metadata.json"),
                    (
                        "n_step=000-class=sklearn.pipeline.Pipeline",
                        OrderedDict(
                            [
                                (
                                    "n_step=000-class=sklearn.decomposition.pca.PCA",
                                    "pca1.pkl.gz",
                                ),
                                (
                                    "n_step=001-class=sklearn.pipeline.FeatureUnion",
                                    "params.json",
                                ),
                                (
                                    "n_step=001-class=sklearn.pipeline.FeatureUnion",
                                    OrderedDict(
                                        [
                                            (
                                                "n_step=000-class=sklearn.decomposition.pca.PCA",
                                                "pca2.pkl.gz",
                                            ),
                                            (
                                                "n_step=001-class=sklearn.pipeline.Pipeline",
                                                OrderedDict(
                                                    [
                                                        (
                                                            "n_step=000-class=sklearn.preprocessing.data.MinMaxScaler",
                                                            "minmax.pkl.gz",
                                                        ),
                                                        (
                                                            "n_step=001-class=sklearn.decomposition.truncated_svd.TruncatedSVD",
                                                            "truncsvd.pkl.gz",
                                                        ),
                                                    ]
                                                ),
                                            ),
                                        ]
                                    ),
                                ),
                                (
                                    "n_step=002-class=gordo_components.model.models.KerasAutoEncoder",
                                    "model.h5",
                                ),
                                (
                                    "n_step=002-class=gordo_components.model.models.KerasAutoEncoder",
                                    "params.json",
                                ),
                            ]
                        ),
                    ),
                ]
            )

            self._structure_verifier(prefix_dir=tmp, structure=expected_structure)

            # Test load from the serialized pipeline above
            pipe_clone = serializer.load(tmp)
            metadata_clone = serializer.load_metadata(tmp)

            # Ensure the metadata was saved and loaded back
            self.assertEqual(metadata, metadata_clone)

            # Verify same state for both pipelines
            y_hat_pipe1 = pipe.transform(X.copy()).flatten()
            y_hat_pipe2 = pipe_clone.transform(X.copy()).flatten()
            self.assertTrue(np.allclose(y_hat_pipe1, y_hat_pipe2))

    def test_dump_load_keras_directly(self):

        model = KerasAutoEncoder(kind="feedforward_symetric")

        X = np.random.random(size=100).reshape(10, 10)
        model.fit(X.copy(), X.copy())

        with TemporaryDirectory() as tmp:
            serializer.dump(model, tmp)

            model_clone = serializer.load(tmp)

            self.assertTrue(
                np.allclose(
                    model.transform(X.copy()).flatten(),
                    model_clone.transform(X.copy()).flatten(),
                )
            )
