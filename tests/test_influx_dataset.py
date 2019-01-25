import os
import unittest
import yaml
import docker
import logging
import time
import dateutil.parser

import pandas as pd
import numpy as np

from influxdb import InfluxDBClient
from click.testing import CliRunner

from gordo_components.dataset._datasets import InfluxBackedDataset
from gordo_components.dataset import get_dataset

import pytest

logger = logging.getLogger(__name__)


SOURCE_DB_NAME = "sensors"  # This is also used for influx measurement name!
DESTINATION_DB_NAME = "destdb"


class PredictionInfluxTestCase(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        for container in cls.containers:
            logger.info(f"Killing container: {container.name}")
            container.kill()

    @classmethod
    def setUpClass(cls):
        cls.runner = CliRunner()
        cls.docker_client = docker.from_env()
        cls.containers = []
        logger.info("Starting up influx!")
        influx = cls.docker_client.containers.run(
            image="influxdb:1.7-alpine",
            environment={
                "INFLUXDB_DB": SOURCE_DB_NAME,
                "INFLUXDB_ADMIN_USER": "root",
                "INFLUXDB_ADMIN_PASSWORD": "root",
            },
            ports={"8086/tcp": "8087"},
            remove=True,
            detach=True,
        )
        cls.containers.append(influx)
        time.sleep(3)  # Give Influx some time to initialize
        logger.info(f"Started influx DB: {influx.name}")
        cls._seed_source_db()
        cls.influx_config = {
            "host": "localhost",
            "port": 8087,
            "username": "root",
            "password": "root",
            "database": SOURCE_DB_NAME,  # this is also the measurement name - to change
            "proxies": {"http": "", "https": ""},
        }

    @staticmethod
    def _load_unique_sensors():
        """
        Loads unique tags from test/data/*config* files
        """
        sensors = set()
        path_to_config = os.path.dirname(os.path.abspath(__file__))
        config = os.path.join(path_to_config, "config-test.yaml")
        with open(config) as f:
            config = yaml.load(f)
        for machine in config["machines"].values():
            for sensor in machine["tags"]:
                sensors.add(sensor)
        return sensors

    @staticmethod
    def _seed_source_db():
        """
        Seed the source db with random, made up tags
        """
        client = InfluxDBClient(
            "localhost",
            8087,
            "root",
            "root",
            SOURCE_DB_NAME,
            proxies={"http": "", "https": ""},
        )
        dates = pd.date_range(
            start="2016-01-01", periods=1440, freq="min"
        )  # Minute intervals for 1 day
        for sensor in PredictionInfluxTestCase._load_unique_sensors():
            logger.info(f"Loading tag: {sensor}")
            points = np.random.random(size=dates.shape[0])
            data = [
                {
                    "measurement": SOURCE_DB_NAME,
                    "tags": {"tag": sensor},
                    "time": f"{date}",
                    "fields": {"Value": point},
                }
                for point, date in zip(points, dates)
            ]
            client.write_points(data)

    @pytest.fixture(autouse=True)
    def caplog_fixture(self, caplog):
        self.caplog = caplog

    def test_read_single_sensor_empty_data_time_range_indexerror(self):

        """
        Asserts that an IndexError is raised because the dates requested are outside the existing time period
        """

        from_ts = "2017-01-01T09:11:00+00:00"
        to_ts = "2017-01-01T10:30:00+00:00"
        from_ts = dateutil.parser.isoparse(from_ts)
        to_ts = dateutil.parser.isoparse(to_ts)
        ds = InfluxBackedDataset(
            influx_config=self.influx_config, from_ts=from_ts, to_ts=to_ts
        )
        tag = "TRC-FIQ -23-0453N"
        with self.caplog.at_level(logging.CRITICAL):
            with self.assertRaises(IndexError):
                ds.read_single_sensor(tag)

    def test_read_single_sensor_empty_data_invalid_tag_name_valueerror(self):
        """
        Asserts that a ValueError is raised because the tag name inputted is invalid
        """

        from_ts = "2016-01-01T09:11:00+00:00"
        to_ts = "2016-01-01T10:30:00+00:00"
        from_ts = dateutil.parser.isoparse(from_ts)
        to_ts = dateutil.parser.isoparse(to_ts)
        ds = InfluxBackedDataset(
            influx_config=self.influx_config, from_ts=from_ts, to_ts=to_ts
        )
        tag = "TRC-FIQ -23-045N"
        with self.assertRaises(ValueError):
            ds.read_single_sensor(tag)

    def test__list_of_tags_from_influx_validate_tag_names(self):
        from_ts = "2016-01-01T09:11:00+00:00"
        to_ts = "2016-01-01T10:30:00+00:00"
        from_ts = dateutil.parser.isoparse(from_ts)
        to_ts = dateutil.parser.isoparse(to_ts)
        ds = InfluxBackedDataset(
            influx_config=self.influx_config, from_ts=from_ts, to_ts=to_ts
        )
        expected_tags = {
            "TRC-FIQ -23-0453N",
            "TRC-FIQ -80-0303N",
            "TRC-FIQ -80-0703N",
            "TRC-FIQ -80-0704N",
            "TRC-FIQ -80-0705N",
        }
        list_of_tags = ds._list_of_tags_from_influx()
        tags = set(list_of_tags)
        self.assertTrue(
            expected_tags == tags,
            msg=f"Expected tags = {expected_tags}" f"outputted {tags}",
        )

    def test_influx_dataset_attrs(self):
        """
        Test expected attributes
        """
        from_ts = "2016-01-01T09:11:00+00:00"
        to_ts = "2016-01-01T10:30:00+00:00"
        from_ts = dateutil.parser.isoparse(from_ts)
        to_ts = dateutil.parser.isoparse(to_ts)
        influx_config = self.influx_config
        tag_list = [
            "TRC-FIQ -23-0453N",
            "TRC-FIQ -80-0303N",
            "TRC-FIQ -80-0703N",
            "TRC-FIQ -80-0704N",
        ]
        config = {
            "type": "InfluxBackedDataset",
            "from_ts": from_ts,
            "to_ts": to_ts,
            "influx_config": influx_config,
            "tag_list": tag_list,
        }
        dataset = get_dataset(config)
        self.assertTrue(hasattr(dataset, "get_metadata"))

        metadata = dataset.get_metadata()
        self.assertTrue(isinstance(metadata, dict))
