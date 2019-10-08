import os
import io
import json
import typing
import re
import time
import logging
from contextlib import contextmanager
from typing import List

import docker
import numpy as np
import pandas as pd

import responses
import requests
from asynctest import mock as async_mock
from influxdb import InfluxDBClient
from flask import Request

from gordo_components.model import models
from gordo_components.watchman import server as watchman_server
from gordo_components.dataset.sensor_tag import SensorTag
from gordo_components.dataset.sensor_tag import to_list_of_strings

logger = logging.getLogger(__name__)

SENSORTAG_LIST = [SensorTag(f"tag-{i}", None) for i in range(4)]
SENSORS_STR_LIST = to_list_of_strings(SENSORTAG_LIST)
INFLUXDB_NAME = "testdb"
INFLUXDB_USER = "root"
INFLUXDB_PASSWORD = "root"
INFLUXDB_MEASUREMENT = "sensors"

INFLUXDB_URI = f"{INFLUXDB_USER}:{INFLUXDB_PASSWORD}@localhost:8086/{INFLUXDB_NAME}"

INFLUXDB_FIXTURE_ARGS = (
    SENSORS_STR_LIST,
    INFLUXDB_NAME,
    INFLUXDB_USER,
    INFLUXDB_PASSWORD,
    SENSORS_STR_LIST,
)

GORDO_HOST = "localhost"
GORDO_PROJECT = "gordo-test"
GORDO_TARGETS = ["machine-1"]
GORDO_SINGLE_TARGET = GORDO_TARGETS[0]


def wait_for_influx(max_wait=30, influx_host="localhost:8086"):
    """Waits up to `max_wait` seconds for influx at `influx_host` to start up.

    Checks by pinging inflix at /ping.

    Parameters
    ----------
    influx_host : str
        Where is influx?
    max_wait : int
        How many seconds to wait in total for influx to start up

    Returns
    -------
    bool
        True if influx started up, False if it did not start during the `max_wait`
        period.

    """
    healtcheck_endpoint = f"http://{influx_host}/ping?verbose=true"
    before_time = time.perf_counter()
    influx_ok = False
    while not influx_ok and time.perf_counter() - before_time < max_wait:
        try:
            code = requests.get(
                healtcheck_endpoint, timeout=1, proxies={"https": "", "http": ""}
            ).status_code
            logger.debug(f"Influx gave code {code}")
            influx_ok = code == 200
        except requests.exceptions.ConnectionError:
            influx_ok = False
        time.sleep(0.5)
    if (time.perf_counter() - before_time) < max_wait:
        logger.info("Found that influx started")
        return True
    else:
        logger.warning("Found that influx never started")
        return False


def _post_patch(*args, **kwargs):
    kwargs = {k: v for k, v in kwargs.items() if k != "session"}

    # Warning, ugly; aiohttp.Session.post calls request's version of 'files' 'data'
    if "data" in kwargs:
        kwargs["files"] = kwargs.pop("data")

    resp = requests.post(*args, **kwargs)
    if resp.headers["Content-Type"] == "application/json":
        return resp.json()
    else:
        return resp.content


def _get_patch(*args, **kwargs):
    resp = requests.get(*args, **{k: v for k, v in kwargs.items() if k != "session"})
    if resp.headers["Content-Type"] == "application/json":
        return resp.json()
    else:
        return resp.content


@contextmanager
def watchman(
    host: str,
    project: str,
    targets: typing.List[str],
    model_location: str,
    namespace: str = "default",
):
    """
    # TODO: This is bananas, make into a proper object with context support?

    Mock a deployed watchman deployment

    Parameters
    ----------
    host: str
        Host watchman should pretend to run on
    project: str
        Project watchman should pretend to care about
    targets:
        Targets watchman should pretend to care about
    model_location: str
        Directory of the model to use in the target(s)
    namespace: str
        Namespace for watchman to make requests in.

    Returns
    -------
    None
    """
    from gordo_components.server import server as gordo_ml_server

    with temp_env_vars(MODEL_COLLECTION_DIR=model_location):
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
                resp = getattr(gordo_server_app, request.method.lower())(
                    request.path_url, **kwargs
                )
                return (
                    200,
                    resp.headers,
                    json.dumps(resp.json) if resp.json is not None else resp.data,
                )

        with responses.RequestsMock(
            assert_all_requests_are_fired=False
        ) as rsps, async_mock.patch(
            "gordo_components.client.io.get", side_effect=_get_patch
        ), async_mock.patch(
            "gordo_components.client.io.post", side_effect=_post_patch
        ):

            # Gordo ML Server requests
            rsps.add_callback(
                responses.GET,
                re.compile(
                    rf".*ambassador.{namespace}.*\/gordo\/v0\/{project}\/.*.\/.*"
                ),
                callback=gordo_ml_server_callback,
                content_type="application/json",
            )
            rsps.add_callback(
                responses.POST,
                re.compile(
                    rf".*ambassador.{namespace}.*\/gordo\/v0\/{project}\/.*.\/.*"
                ),
                callback=gordo_ml_server_callback,
                content_type="application/json",
            )
            rsps.add_callback(
                responses.GET,
                re.compile(rf".*{host}.*\/gordo\/v0\/{project}\/.*.\/.*"),
                callback=gordo_ml_server_callback,
                content_type="application/json",
            )
            rsps.add_callback(
                responses.POST,
                re.compile(rf".*{host}.*\/gordo\/v0\/{project}\/.*.\/.*"),
                callback=gordo_ml_server_callback,
                content_type="application/json",
            )

            rsps.add_passthru("http+docker://")  # Docker
            rsps.add_passthru("http://localhost:8086")  # Local influx
            rsps.add_passthru("http://localhost:8087")  # Local influx

            # Create a watchman test app
            watchman_app = watchman_server.build_app(
                project_name=project,
                project_version="v123",
                target_names=targets,
                namespace=namespace,
                ambassador_host=host,
                listen_to_kubernetes=False,
            )
            watchman_app.testing = True
            watchman_app = watchman_app.test_client()

            def watchman_callback(_request):
                """
                Redirect calls to a gordo endpoint to reflect what the local testing app gives
                """
                headers = {}
                resp = watchman_app.get("/").json
                return 200, headers, json.dumps(resp)

            # Watchman requests
            rsps.add_callback(
                responses.GET,
                re.compile(rf".*{host}.*\/gordo\/v0\/{project}\/$"),
                callback=watchman_callback,
                content_type="application/json",
            )
            yield


@contextmanager
def temp_env_vars(**kwargs):
    """
    Temporarily set the process environment variables
    """
    _env = os.environ.copy()

    for key in kwargs:
        os.environ[key] = kwargs[key]

    yield

    os.environ.clear()
    os.environ.update(_env)


@contextmanager
def influxdatabase(
    sensors: List[SensorTag], db_name: str, user: str, password: str, measurement: str
):
    """
    Setup a docker based InfluxDB with data points from 2016-01-1 until 2016-01-02 by minute

    Returns
    -------
    InfluxDB
        An interface to the running db instance with .reset() for convenient resetting of the
        db to it's default state, (with original sensors)
    """

    client = docker.from_env()

    logger.info("Starting up influx!")
    influx = None
    try:
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
        if not wait_for_influx(influx_host="localhost:8086"):
            raise TimeoutError("Influx failed to start")

        logger.info(f"Started influx DB: {influx.name}")

        # Create the interface to the running instance, set default state, and yield it.
        db = InfluxDB(sensors, db_name, user, password, measurement)
        db.reset()
        logger.info("STARTED INFLUX INSTANCE")
        yield db

    finally:
        logger.info("Killing influx container")
        if influx:
            influx.kill()
        logger.info("Killed influx container")


class InfluxDB:
    """
    Simple interface to a running influx.
    """

    def __init__(
        self,
        sensors: List[SensorTag],
        db_name: str,
        user: str,
        password: str,
        measurement: str,
    ):
        self.sensors = sensors
        self.db_name = db_name
        self.user = user
        self.password = password
        self.measurement = measurement

    def reset(self):
        """
        Set the db to contain the default data
        """
        # Seed database with some records
        influx_client = InfluxDBClient(
            "localhost",
            8086,
            self.user,
            self.password,
            self.db_name,
            proxies={"http": "", "https": ""},
        )

        # Drop and re-create the database
        influx_client.drop_database(self.db_name)
        influx_client.create_database(self.db_name)

        dates = pd.date_range(
            start="2016-01-01", periods=2880, freq="min"
        )  # Minute intervals for 2 days

        logger.info("Seeding database")
        for sensor in self.sensors:
            logger.info(f"Loading tag: {sensor.name}")
            points = np.random.random(size=dates.shape[0])
            data = [
                {
                    "measurement": self.measurement,
                    "tags": {"tag": sensor.name},
                    "time": f"{date}",
                    "fields": {"Value": point},
                }
                for point, date in zip(points, dates)
            ]
            influx_client.write_points(data)


def get_model(config):
    type = config.get("type", "")
    Model = getattr(models, type, None)
    if Model is None:
        raise ValueError(
            f'Type of model: "{type}" either not provided or not supported'
        )
    return Model(**config)
