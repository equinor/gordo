# -*- coding: utf-8 -*-

import asyncio
import os
import logging
import tempfile
from typing import List
from unittest.mock import patch

import pytest

from gordo_components import serializer
from gordo_components.machine.dataset.sensor_tag import SensorTag
from gordo_components.machine.dataset.data_provider.providers import InfluxDataProvider
from gordo_components.server import server
from gordo_components.builder.local_build import local_build
from gordo_components.machine.dataset import sensor_tag

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
    Fixture: Temporary directory, removed after test completion
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture(scope="session")
def gordo_name():
    # One gordo-target name
    return tu.GORDO_TARGETS[0]


@pytest.fixture(scope="session")
def gordo_project():
    return tu.GORDO_PROJECT


@pytest.fixture(scope="session")
def gordo_revision():
    return tu.GORDO_REVISION


@pytest.fixture(scope="session")
def api_version():
    return "v0"


@pytest.fixture(scope="session")
def base_route(api_version, gordo_project, gordo_name):
    return f"/gordo/{api_version}/{gordo_project}/{gordo_name}"


@pytest.fixture(scope="session")
def model_collection_directory(gordo_revision: str):
    with tempfile.TemporaryDirectory() as tmp_dir:
        collection_dir = os.path.join(tmp_dir, gordo_revision)
        os.makedirs(collection_dir, exist_ok=True)
        yield collection_dir


@pytest.fixture(scope="session")
def config_str(gordo_name: str, sensors: List[SensorTag]):
    """
    Fixture: Default config for testing
    """
    return f"""
            machines:
              - dataset:
                  tags:
                    - {sensors[0].name}
                    - {sensors[1].name}
                    - {sensors[2].name}
                    - {sensors[3].name}
                  target_tag_list:
                    - {sensors[0].name}
                    - {sensors[1].name}
                    - {sensors[2].name}
                    - {sensors[3].name}
                  train_start_date: '2019-01-01T00:00:00+00:00'
                  train_end_date: '2019-10-01T00:00:00+00:00'
                  asset: asgb
                  data_provider:
                    type: RandomDataProvider
                metadata:
                  information: Some sweet information about the model
                model:
                  gordo_components.machine.model.anomaly.diff.DiffBasedAnomalyDetector:
                    require_thresholds: false
                    base_estimator:
                      sklearn.pipeline.Pipeline:
                        steps:
                        - sklearn.preprocessing.data.MinMaxScaler
                        - gordo_components.machine.model.models.KerasAutoEncoder:
                            kind: feedforward_hourglass
                name: {gordo_name}
             """


@pytest.fixture(scope="session")
def trained_model_directory(
    model_collection_directory: str, config_str: str, gordo_name: str
):
    """
    Fixture: Train a basic AutoEncoder and save it to a given directory
    will also save some metadata with the model
    """

    # Model specific to the model being trained here
    model_dir = os.path.join(model_collection_directory, gordo_name)
    os.makedirs(model_dir, exist_ok=True)

    builder = local_build(config_str=config_str, enable_mlflow=False)
    model, metadata = next(builder)  # type: ignore
    serializer.dump(model, model_dir, metadata=metadata.to_dict())
    yield model_dir


@pytest.fixture
def metadata(trained_model_directory):
    return serializer.load_metadata(trained_model_directory)


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
def gordo_ml_server_client(
    request, model_collection_directory, trained_model_directory
):

    with tu.temp_env_vars(MODEL_COLLECTION_DIR=model_collection_directory):

        app = server.build_app()
        app.testing = True

        # always return a valid asset for any tag name
        with patch.object(sensor_tag, "_asset_from_tag_name", return_value="default"):
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
def ml_server(
    model_collection_directory,
    trained_model_directory,
    host=tu.GORDO_HOST,
    project=tu.GORDO_PROJECT,
    targets=tu.GORDO_TARGETS,
):
    with tu.ml_server_deployment(
        host=host,
        project=project,
        targets=targets,
        model_location=model_collection_directory,
    ):
        # always return a valid asset for any tag name
        with patch.object(sensor_tag, "_asset_from_tag_name", return_value="default"):
            yield


@pytest.fixture(scope="session")
def repo_dir():
    """
    Return the repository directory for gordo infrastructure
    """
    return os.path.join(os.path.dirname(__file__), "..")
