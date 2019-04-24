# -*- coding: utf-8 -*-

import asyncio
import logging
import tempfile
import typing

import ruamel
import pytest
import numpy as np

from gordo_components import serializer
from tests.gordo_components.server.test_gordo_server import SENSORS, influxdatabase
from tests.utils import watchman


logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def check_event_loop():
    loop = asyncio.get_event_loop()
    if loop.is_closed():
        logger.critical("Creating new event loop!")
        asyncio.set_event_loop(asyncio.new_event_loop())


@pytest.fixture(scope="session")
def sensors():
    return SENSORS


@pytest.fixture
def tmp_dir():
    """
    Fixture: Temporary directory
    """
    return tempfile.TemporaryDirectory()


@pytest.yield_fixture(scope="session")
def trained_model_directory(sensors: typing.List[str]):
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


@pytest.yield_fixture(scope="module")
def influxdb(
    sensors, db_name="testdb", user="root", password="root", measurement="sensors"
):
    with influxdatabase(
        sensors=sensors,
        db_name=db_name,
        user=user,
        password=password,
        measurement=measurement,
    ):
        yield


@pytest.yield_fixture(scope="module")
def watchman_service(
    trained_model_directory,
    host="localhost",
    project="gordo-test",
    targets=["machine-1"],
):
    with watchman(
        host=host,
        project=project,
        targets=targets,
        model_location=trained_model_directory,
    ):
        yield
