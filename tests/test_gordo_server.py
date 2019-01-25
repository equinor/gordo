# -*- coding: utf-8 -*-

import unittest
import logging
import tempfile

import ruamel.yaml
import numpy as np

from gordo_components.server import server
from gordo_components import serializer
from tests.utils import temp_env_vars

logger = logging.getLogger(__name__)


class GordoServerTestCase(unittest.TestCase):
    """
    Test expected functionality of the gordo server
    """

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls._build_model(cls.tmpdir.name)

    @staticmethod
    def _build_model(target_dir):
        definition = ruamel.yaml.load(
            """
            sklearn.pipeline.Pipeline:
                steps:
                    - sklearn.preprocessing.data.MinMaxScaler
                    - gordo_components.model.models.KerasAutoEncoder:
                        kind: feedforward_symetric
                memory:
            """,
            Loader=ruamel.yaml.Loader,
        )
        model = serializer.pipeline_from_definition(definition)
        X = np.random.random(size=100).reshape(-10, 10)
        model.fit(X, X)
        serializer.dump(model, target_dir, metadata={"model-name": "test-model"})

    def setUp(self):
        with temp_env_vars(MODEL_LOCATION=self.tmpdir.name):
            app = server.build_app()
            app.testing = True
            self.app = app.test_client()

    def test_healthcheck_endpoint(self):
        """
        Test expected behavior of /healthcheck
        """
        with temp_env_vars(MODEL_LOCATION=self.tmpdir.name):
            resp = self.app.get("/healthcheck")
        self.assertEqual(resp.status_code, 200)

        data = resp.get_json()
        logger.debug(f"Got resulting JSON response: {data}")
        self.assertTrue("version" in data)

    def test_metadata_endpoint(self):
        """
        Test the expected behavior of /metadata
        """
        with temp_env_vars(MODEL_LOCATION=self.tmpdir.name):
            resp = self.app.get("/metadata")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue("model-metadata" in data)
        self.assertEqual(data["model-metadata"]["model-name"], "test-model")

    def test_predictions_endpoint(self):
        """
        Test the expected behavior of /predictions
        """
        with temp_env_vars(MODEL_LOCATION=self.tmpdir.name):

            # This should give an error, input data is not the same as data trained with
            resp = self.app.post("/predictions", json={"X": [[1, 2, 3], [1, 2, 3]]})
            self.assertEqual(resp.status_code, 400)
            data = resp.get_json()
            logger.debug(f"Got resulting JSON response: {data}")
            self.assertTrue("error" in data)

            # These should be fine; multi-record and single record prediction requests.
            for data in [
                np.random.random(size=20).reshape(2, 10).tolist(),
                np.random.random(size=10).tolist(),
            ]:

                resp = self.app.post("/predictions", json={"X": data})
                self.assertEqual(resp.status_code, 200)
                data = resp.get_json()
                logger.debug(f"Got resulting JSON response: {data}")
                self.assertTrue("output" in data)
                np.asanyarray(data["output"])

            # Should fail with 400 if no X is supplied.
            resp = self.app.post("/predictions", json={"no-x-here": True})
            self.assertEqual(resp.status_code, 400)

            # Providing mismatching record lengths should cause 400
            resp = self.app.post("/predictions", json={"X": [[1, 2, 3], [1, 2]]})
            self.assertEqual(resp.status_code, 400)
