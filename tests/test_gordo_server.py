# -*- coding: utf-8 -*-

import unittest
import logging
import tempfile
import time
from contextlib import contextmanager

import pytest
import docker
import ruamel.yaml
import numpy as np
import pandas as pd

from influxdb import InfluxDBClient

from gordo_components.server import server
from gordo_components import serializer
from gordo_components.data_provider.providers import InfluxDataProvider
from tests.utils import temp_env_vars

logger = logging.getLogger(__name__)


SENSORS = [f"tag-{i}" for i in range(10)]
INFLUX_DB = "sensors"
INFLUX_ADMIN_USER = "root"
INFLUX_ADMIN_PASSWORD = "root"


@contextmanager
def influxdatabase():
    """
    Setup a docker based InfluxDB
    """

    client = docker.from_env()

    logger.info("Starting up influx!")
    influx = client.containers.run(
        image="influxdb:1.7-alpine",
        environment={
            "INFLUXDB_DB": INFLUX_DB,
            "INFLUXDB_ADMIN_USER": INFLUX_ADMIN_USER,
            "INFLUXDB_ADMIN_PASSWORD": INFLUX_ADMIN_PASSWORD,
        },
        ports={"8086/tcp": "8086"},
        remove=True,
        detach=True,
    )
    time.sleep(4)  # Give Influx some time to initialize
    logger.info(f"Started influx DB: {influx.name}")

    # Seed database with some records
    influx_client = InfluxDBClient(
        "localhost",
        8086,
        INFLUX_ADMIN_USER,
        INFLUX_ADMIN_PASSWORD,
        INFLUX_DB,
        proxies={"http": "", "https": ""},
    )
    dates = pd.date_range(
        start="2016-01-01", periods=2880, freq="min"
    )  # Minute intervals for 2 days

    logger.info("Seeding database")
    for sensor in SENSORS:
        logger.info(f"Loading tag: {sensor}")
        points = np.random.random(size=dates.shape[0])
        data = [
            {
                "measurement": INFLUX_DB,
                "tags": {"tag": sensor},
                "time": f"{date}",
                "fields": {"Value": point},
            }
            for point, date in zip(points, dates)
        ]
        influx_client.write_points(data)

    yield

    logger.info("Killing influx container")
    influx.kill()
    logger.info("Killed influx container")


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
                        kind: feedforward_hourglass
                memory:
            """,
            Loader=ruamel.yaml.Loader,
        )
        model = serializer.pipeline_from_definition(definition)
        X = np.random.random(size=100).reshape(-10, 10)
        model.fit(X, X)
        serializer.dump(
            model,
            target_dir,
            metadata={
                "dataset": {"tag_list": SENSORS, "resolution": "10T"},
                "user-defined": {"model-name": "test-model"},
            },
        )

    def setUp(self):
        with temp_env_vars(MODEL_LOCATION=self.tmpdir.name):
            provider = InfluxDataProvider(
                measurement=INFLUX_DB,
                value_name="Value",
                proxies={"https": "", "http": ""},
                database=INFLUX_DB,
            )
            app = server.build_app(data_provider=provider)
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
        self.assertTrue("gordo-server-version" in data)

    def test_metadata_endpoint(self):
        """
        Test the expected behavior of /metadata
        """
        with temp_env_vars(MODEL_LOCATION=self.tmpdir.name):
            resp = self.app.get("/metadata")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue("metadata" in data)
        self.assertEqual(data["metadata"]["user-defined"]["model-name"], "test-model")

    def test_prediction_endpoint_post(self):
        """
        Test the expected behavior of /predictions
        """
        with temp_env_vars(MODEL_LOCATION=self.tmpdir.name):

            # This should give an error, input data is not the same as data trained with
            resp = self.app.post("/prediction", json={"X": [[1, 2, 3], [1, 2, 3]]})
            self.assertEqual(resp.status_code, 400)
            data = resp.get_json()
            logger.debug(f"Got resulting JSON response: {data}")
            self.assertTrue("error" in data)

            # These should be fine; multi-record and single record prediction requests.
            for data in [
                np.random.random(size=20).reshape(2, 10).tolist(),
                np.random.random(size=10).tolist(),
            ]:

                resp = self.app.post("/prediction", json={"X": data})
                self.assertEqual(resp.status_code, 200)
                data = resp.get_json()
                logger.debug(f"Got resulting JSON response: {data}")
                self.assertTrue("output" in data)
                np.asanyarray(data["output"])

            # Should fail with 400 if no X is supplied.
            resp = self.app.post("/prediction", json={"no-x-here": True})
            self.assertEqual(resp.status_code, 400)

            # Providing mismatching record lengths should cause 400
            resp = self.app.post("/prediction", json={"X": [[1, 2, 3], [1, 2]]})
            self.assertEqual(resp.status_code, 400)

    def test_download_model(self):
        """
        Test we can download a model, loadable via serializer.loads()
        """
        with temp_env_vars(MODEL_LOCATION=self.tmpdir.name):
            resp = self.app.get("/download-model")

        serialized_model = resp.get_data()
        model = serializer.loads(serialized_model)

        # All models have a fit method
        self.assertTrue(hasattr(model, "fit"))

        # Models MUST have either predict or transform
        self.assertTrue(hasattr(model, "predict") or hasattr(model, "transform"))

    @influxdatabase()
    @pytest.mark.dockertest
    def test_prediction_endpoint_get(self):
        """
        Client can ask for a timerange based prediction
        """
        with temp_env_vars(MODEL_LOCATION=self.tmpdir.name):
            resp = self.app.get(
                "/prediction",
                json={
                    "start": "2016-01-01T00:00:00+00:00",
                    "end": "2016-01-01T12:00:00+00:00",
                },
            )

        self.assertTrue(resp.status_code, 200)
        self.assertTrue("output" in resp.json)

        # Verify keys & structure of one output record output contains a list of:
        # {'start': timestamp, 'end': timestamp, 'tags': {'tag': float}, 'total_abnormality': float}
        self.assertTrue("start" in resp.json["output"][0])
        self.assertTrue("end" in resp.json["output"][0])
        self.assertTrue("tags" in resp.json["output"][0])
        self.assertIsInstance(resp.json["output"][0]["tags"], dict)
        self.assertTrue("total_anomaly" in resp.json["output"][0])

        # Request greater than 1 day should be a bad request
        with temp_env_vars(MODEL_LOCATION=self.tmpdir.name):
            resp = self.app.get(
                "/prediction",
                json={
                    "start": "2016-01-01T00:00:00+00:00",
                    "end": "2016-01-03T12:00:00+00:00",
                },
            )
            self.assertTrue(resp.status_code, 400)
            self.assertTrue("error" in resp.json)

        # Requests for overlapping time sample buckets should only produce
        # predictions whose bucket falls completely within the requested
        # time range
        with temp_env_vars(MODEL_LOCATION=self.tmpdir.name):

            # This should give one prediction with start end of
            # 2016-01-01 00:10:00+00:00 and 2016-01-01 00:20:00+00:00, respectively.
            # because it will not go over, ie. giving a bucket from 00:20:00 to 00:30:00
            # but can give a bucket which contains data before the requested start date.
            resp = self.app.get(
                "/prediction",
                json={
                    "start": "2016-01-01T00:11:00+00:00",
                    "end": "2016-01-01T00:21:00+00:00",
                },
            )
            self.assertTrue(resp.status_code, 200)
            self.assertEqual(
                len(resp.json["output"]),
                1,
                msg=f"Expected one prediction, got: {resp.json}",
            )
            self.assertEqual(
                resp.json["output"][0]["start"], "2016-01-01 00:10:00+00:00"
            )
            self.assertEqual(resp.json["output"][0]["end"], "2016-01-01 00:20:00+00:00")
