# -*- coding: utf-8 -*-

import asyncio
import io
import json
import logging
import os
import re
import tempfile
from threading import Lock
from typing import List
from unittest.mock import patch

import docker
from flask import Request
import pytest
import responses

from gordo import serializer
from gordo.machine.dataset.sensor_tag import SensorTag

from gordo.server import server
from gordo.builder.local_build import local_build
from gordo.machine.dataset import sensor_tag
from gordo.machine.dataset.sensor_tag import to_list_of_strings
from gordo.server import server as gordo_ml_server

from tests import utils as tu

logger = logging.getLogger(__name__)

TEST_SERVER_MUTEXT = Lock()


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
def gordo_host():
    return "localhost"


@pytest.fixture(scope="session")
def gordo_project():
    return "gordo-test"


@pytest.fixture(scope="session")
def gordo_name():
    return "machine-1"


@pytest.fixture(scope="session")
def gordo_single_target(gordo_name):
    return gordo_name


@pytest.fixture(scope="session")
def gordo_targets(gordo_single_target):
    return [gordo_single_target]


@pytest.fixture(scope="session")
def sensors():
    return [SensorTag(f"tag-{i}", None) for i in range(4)]


@pytest.fixture(scope="session")
def sensors_str(sensors):
    return to_list_of_strings(sensors)


@pytest.fixture(scope="session")
def influxdb_name():
    return "testdb"


@pytest.fixture(scope="session")
def influxdb_user():
    return "root"


@pytest.fixture(scope="session")
def influxdb_password():
    return "root"


@pytest.fixture(scope="session")
def influxdb_measurement():
    return "sensors"


@pytest.fixture(scope="session")
def influxdb_fixture_args(sensors_str, influxdb_name, influxdb_user, influxdb_password):
    return (sensors_str, influxdb_name, influxdb_user, influxdb_password, sensors_str)


@pytest.fixture(scope="session")
def influxdb_uri(influxdb_user, influxdb_password, influxdb_name):
    return f"{influxdb_user}:{influxdb_password}@localhost:8086/{influxdb_name}"


@pytest.fixture(scope="session")
def gordo_revision():
    return "1234"


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
                  gordo.machine.model.anomaly.diff.DiffBasedAnomalyDetector:
                    require_thresholds: false
                    base_estimator:
                      sklearn.pipeline.Pipeline:
                        steps:
                        - sklearn.preprocessing.data.MinMaxScaler
                        - gordo.machine.model.models.KerasAutoEncoder:
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

    builder = local_build(config_str=config_str)
    model, metadata = next(builder)  # type: ignore
    serializer.dump(model, model_dir, metadata=metadata.to_dict())
    yield model_dir


@pytest.fixture
def metadata(trained_model_directory):
    return serializer.load_metadata(trained_model_directory)


@pytest.fixture(scope="session")
def gordo_ml_server_client(
    request, model_collection_directory, trained_model_directory
):

    with tu.temp_env_vars(MODEL_COLLECTION_DIR=model_collection_directory):

        app = server.build_app()
        app.testing = True

        # always return a valid asset for any tag name
        with patch.object(sensor_tag, "_asset_from_tag_name", return_value="default"):
            yield app.test_client()


@pytest.yield_fixture
def postgresdb():
    client = docker.from_env()
    postgres = client.containers.run(
        image="postgres:11-alpine",
        environment={"POSTGRES_USER": "postgres", "POSTGRES_PASSWORD": "postgres"},
        ports={"5432/tcp": "5432"},
        remove=True,
        detach=True,
    )
    import time

    time.sleep(5)
    yield
    postgres.kill()


@pytest.fixture(scope="session")
def base_influxdb(
    sensors, influxdb_name, influxdb_user, influxdb_password, influxdb_measurement
):
    """
    Fixture to yield a running influx container and pass a tests.utils.InfluxDB
    object which can be used to reset the db to it's original data state.
    """
    client = docker.from_env()

    logger.info("Starting up influx!")
    influx = None
    try:
        influx = client.containers.run(
            image="influxdb:1.7-alpine",
            environment={
                "INFLUXDB_DB": influxdb_name,
                "INFLUXDB_ADMIN_USER": influxdb_user,
                "INFLUXDB_ADMIN_PASSWORD": influxdb_password,
            },
            ports={"8086/tcp": "8086"},
            remove=True,
            detach=True,
        )
        if not tu.wait_for_influx(influx_host="localhost:8086"):
            raise TimeoutError("Influx failed to start")

        logger.info(f"Started influx DB: {influx.name}")

        # Create the interface to the running instance, set default state, and yield it.
        db = tu.InfluxDB(
            sensors,
            influxdb_name,
            influxdb_user,
            influxdb_password,
            influxdb_measurement,
        )
        db.reset()
        logger.info("STARTED INFLUX INSTANCE")
        yield db

    finally:
        logger.info("Killing influx container")
        if influx:
            influx.kill()
        logger.info("Killed influx container")


@pytest.fixture
def influxdb(base_influxdb):
    """
    Fixture to take a running influx and do a reset after each test to ensure
    the data state is the same for each test.
    """
    logger.info("DOING A RESET ON INFLUX DATA")
    base_influxdb.reset()


@pytest.fixture(scope="session")
def argo_version(repo_dir):
    with open(os.path.join(repo_dir, "Dockerfile-GordoDeploy")) as f:
        match = next(re.finditer(r'ARGO_VERSION="(\w\d+.\d+.\d+)"', f.read()), None)
    if match is None:
        raise LookupError(
            "Failed to determine argo version from Dockerfile-GordoDeploy"
        )
    return match.groups()[0]


@pytest.fixture(scope="module")
def ml_server(
    model_collection_directory, trained_model_directory, gordo_host, gordo_project
):
    """
    # TODO: This is bananas, make into a proper object with context support?

    Mock a deployed controller deployment

    Parameters
    ----------
    gordo_host: str
        Host controller should pretend to run on
    gordo_project: str
        Project controller should pretend to care about
    model_collection_directory: str
        Directory of the model to use in the target(s)

    Returns
    -------
    None
    """
    with tu.temp_env_vars(MODEL_COLLECTION_DIR=model_collection_directory):
        # Create gordo ml servers
        gordo_server_app = gordo_ml_server.build_app()
        gordo_server_app.testing = True
        gordo_server_app = gordo_server_app.test_client()

        def gordo_ml_server_callback(request):
            """
            Redirect calls to a gordo server to reflect what the local testing app gives
            will call the correct path (assuminng only single level paths) on the
            gordo app.
            """
            if request.method in ("GET", "POST"):

                kwargs = dict()
                if request.body:
                    flask_request = Request.from_values(
                        content_length=len(request.body),
                        input_stream=io.BytesIO(request.body),
                        content_type=request.headers["Content-Type"],
                        method=request.method,
                    )
                    if flask_request.json:
                        kwargs["json"] = flask_request.json
                    else:
                        kwargs["data"] = {
                            k: (io.BytesIO(f.read()), f.filename)
                            for k, f in flask_request.files.items()
                        }

                with TEST_SERVER_MUTEXT:
                    resp = getattr(gordo_server_app, request.method.lower())(
                        request.path_url, headers=dict(request.headers), **kwargs
                    )
                return (
                    resp.status_code,
                    resp.headers,
                    json.dumps(resp.json) if resp.json is not None else resp.data,
                )

        with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
            rsps.add_callback(
                responses.GET,
                re.compile(rf".*{gordo_host}.*\/gordo\/v0\/{gordo_project}\/.+"),
                callback=gordo_ml_server_callback,
                content_type="application/json",
            )
            rsps.add_callback(
                responses.POST,
                re.compile(rf".*{gordo_host}.*\/gordo\/v0\/{gordo_project}\/.*.\/.*"),
                callback=gordo_ml_server_callback,
                content_type="application/json",
            )

            rsps.add_passthru("http+docker://")  # Docker
            rsps.add_passthru("http://localhost:8086")  # Local influx
            rsps.add_passthru("http://localhost:8087")  # Local influx

            yield


@pytest.fixture(scope="session")
def repo_dir():
    """
    Return the repository directory for gordo infrastructure
    """
    return os.path.join(os.path.dirname(__file__), "..")
