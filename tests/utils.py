import os
import time
import logging
from contextlib import contextmanager
from threading import Lock
from typing import List

import numpy as np
import pandas as pd

import requests
from influxdb import InfluxDBClient


from gordo.machine.model import models
from gordo_dataset.sensor_tag import SensorTag

logger = logging.getLogger(__name__)


TEST_SERVER_MUTEXT = Lock()


def wait_for_influx(max_wait=120, influx_host="localhost:8086"):
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
