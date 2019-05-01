# -*- coding: utf-8 -*-

import asyncio
import logging
import tempfile

import ruamel.yaml
import pytest
import numpy as np
from typing import List

from gordo_components.server import server
from gordo_components import serializer
from gordo_components.data_provider.providers import InfluxDataProvider
from gordo_components.dataset.sensor_tag import SensorTag
from tests import utils as tu

logger = logging.getLogger(__name__)


def pytest_collection_modifyitems(items):
    """
    Update all tests which use influxdb to be marked as a dockertest
    """
    for item in items:
        if hasattr(item, "fixturenames") and "influxdb" in item.fixturenames:
            item.add_marker(pytest.mark.dockertest)


@pytest.fixture(autouse=True)
def check_event_loop():
    loop = asyncio.get_event_loop()
    if loop.is_closed():
        logger.info("Creating new event loop!")
        asyncio.set_event_loop(asyncio.new_event_loop())


@pytest.fixture(scope="session")
def sensors():
    return tu.SENSORTAG_LIST


@pytest.fixture
def tmp_dir():
    """
    Fixture: Temporary directory
    """
    return tempfile.TemporaryDirectory()


@pytest.fixture(scope="session")
def trained_model_directory(sensors: List[SensorTag]):
    """
    Fixture: Train a basic AutoEncoder and save it to a given directory
    will also save some metadata with the model
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
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
        X = np.random.random(size=len(sensors) * 10).reshape(10, len(sensors))
        model.fit(X, X)
        serializer.dump(
            model,
            tmp_dir,
            metadata={
                "dataset": {"tag_list": sensors, "resolution": "10T"},
                "user-defined": {
                    "model-name": "test-model",
                    "machine-name": "machine-1",
                },
            },
        )
        yield tmp_dir


@pytest.fixture(
    # Data Provider(s) per test requiring this client
    params=[
        InfluxDataProvider(
            measurement="sensors",
            value_name="Value",
            proxies={"https": "", "http": ""},
            uri=tu.INFLUXDB_URI,
        )
    ],
    scope="session",
)
def gordo_ml_server_client(request, trained_model_directory):

    with tu.temp_env_vars(MODEL_LOCATION=trained_model_directory):

        app = server.build_app(data_provider=request.param)
        app.testing = True

        yield app.test_client()


@pytest.fixture(scope="session")
def base_influxdb(
    sensors,
    db_name=tu.INFLUXDB_NAME,
    user=tu.INFLUXDB_USER,
    password=tu.INFLUXDB_PASSWORD,
    measurement=tu.INFLUXDB_MEASUREMENT,
):
    """
    Fixture to yield a running influx container and pass a test.utils.InfluxDB
    object which can be used to reset the db to it's original data state.
    """
    with tu.influxdatabase(
        sensors=sensors,
        db_name=db_name,
        user=user,
        password=password,
        measurement=measurement,
    ) as db:
        yield db


@pytest.fixture
def influxdb(base_influxdb):
    """
    Fixture to take a running influx and do a reset after each test to ensure
    the data state is the same for each test.
    """
    logger.info("DOING A RESET ON INFLUX DATA")
    base_influxdb.reset()


@pytest.fixture(scope="module")
def watchman_service(
    trained_model_directory,
    host=tu.GORDO_HOST,
    project=tu.GORDO_PROJECT,
    targets=tu.GORDO_TARGETS,
):
    with tu.watchman(
        host=host,
        project=project,
        targets=targets,
        model_location=trained_model_directory,
    ):
        yield
