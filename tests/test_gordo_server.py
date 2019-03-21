# -*- coding: utf-8 -*-

import unittest
import logging
import tempfile
import time
from contextlib import contextmanager
from typing import List

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


SENSORS = [f"tag-{i}" for i in range(4)]
INFLUX_DB = "sensors"
INFLUX_ADMIN_USER = "root"
INFLUX_ADMIN_PASSWORD = "root"


@contextmanager
def influxdatabase(
    sensors: List[str],
    db_name: str,
    user: str,
    password: str,
    measurement: str = INFLUX_DB,
):
    """
    Setup a docker based InfluxDB with data points from 2016-01-1 until 2016-01-02 by minute
    """

    client = docker.from_env()

    logger.info("Starting up influx!")
    influx = client.containers.run(
        image="influxdb:1.7-alpine",
        environment={
            "INFLUXDB_DB": db_name,
            "INFLUXDB_ADMIN_USER": user,
            "INFLUXDB_ADMIN_PASSWORD": password,
        },
        ports={"8086/tcp": "8086"},
        remove=True,
        detach=True,
    )
    time.sleep(2)  # Give Influx some time to initialize
    logger.info(f"Started influx DB: {influx.name}")

    # Seed database with some records
    influx_client = InfluxDBClient(
        "localhost", 8086, user, password, db_name, proxies={"http": "", "https": ""}
    )
    dates = pd.date_range(
        start="2016-01-01", periods=2880, freq="min"
    )  # Minute intervals for 2 days

    logger.info("Seeding database")
    for sensor in sensors:
        logger.info(f"Loading tag: {sensor}")
        points = np.random.random(size=dates.shape[0])
        data = [
            {
                "measurement": measurement,
                "tags": {"tag": sensor},
                "time": f"{date}",
                "fields": {"Value": point},
            }
            for point, date in zip(points, dates)
        ]
        influx_client.write_points(data)
    try:
        yield
    finally:
        logger.info("Killing influx container")
        influx.kill()
        logger.info("Killed influx container")


class GordoServerBaseTestCase(unittest.TestCase):
    """
    GordoServerBaseTestCase provides the setup of training and serving a model
    via the Gordo ML Server

    The model (nothing special) is a simple auto encoder
    which takes 10 features as input.

    Can use `self.app` as a reference to the Flask testing app instance
    """

    measurement = INFLUX_DB
    database = INFLUX_DB
    username = INFLUX_ADMIN_USER
    password = INFLUX_ADMIN_PASSWORD
    sensors = SENSORS
    tmpdir = tempfile.TemporaryDirectory()

    @classmethod
    def setUpClass(cls):
        cls._build_model(cls.tmpdir.name, cls.sensors)

    @staticmethod
    def _build_model(target_dir: str, sensors: List[str]):
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
        X = np.random.random(size=40).reshape(-1, 4)
        model.fit(X, X)
        serializer.dump(
            model,
            target_dir,
            metadata={
                "dataset": {"tag_list": sensors, "resolution": "10T"},
                "user-defined": {
                    "model-name": "test-model",
                    "machine-name": "machine-1",
                },
            },
        )

    def setUp(self):
        with temp_env_vars(MODEL_LOCATION=self.tmpdir.name):
            provider = InfluxDataProvider(
                measurement=self.measurement,
                value_name="Value",
                proxies={"https": "", "http": ""},
                database=self.database,
                username=self.username,
                password=self.password,
            )
            app = server.build_app(data_provider=provider)
            app.testing = True
            self.app = app.test_client()


class GordoServerTestCase(GordoServerBaseTestCase):
    """
    Test expected functionality of the gordo server
    """

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
                np.random.random(size=len(SENSORS) * 10)
                .reshape((10, len(SENSORS)))
                .tolist(),
                np.random.random(size=len(SENSORS)).tolist(),
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

    @pytest.mark.dockertest
    @influxdatabase(
        sensors=GordoServerBaseTestCase.sensors,
        db_name=GordoServerBaseTestCase.database,
        user=GordoServerBaseTestCase.username,
        password=GordoServerBaseTestCase.password,
        measurement=GordoServerBaseTestCase.measurement,
    )
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

        self.assertEqual(resp.status_code, 200)
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
