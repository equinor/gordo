# -*- coding: utf-8 -*-

import pytest
import logging

from typing import List
from unittest.mock import patch, MagicMock

import pandas as pd
import numpy as np

from gordo.client.forwarders import ForwardPredictionsIntoInflux
from gordo.client.utils import influx_client_from_uri
from gordo.machine import Machine
from gordo_dataset import sensor_tag


@pytest.fixture
def mock_influx_dataframe_client():
    mock = MagicMock()
    return mock


def test_write_to_influx_with_retries(mock_influx_dataframe_client, caplog):
    def find_caplog_record(levelname, msg):
        for record in caplog.records:
            if record.levelname == levelname and record.msg == msg:
                return True
        return False

    with patch.object(
        ForwardPredictionsIntoInflux, "_stack_to_name_value_columns"
    ) as _stack_to_name_value_columns, patch("time.sleep"), caplog.at_level(
        logging.INFO
    ):
        _stack_to_name_value_columns.side_effect = lambda v: v
        df = pd.DataFrame()
        mock_influx_dataframe_client.write_points.side_effect = OSError(
            "Connection refused"
        )
        forwarder = ForwardPredictionsIntoInflux(
            destination_influx_uri="root:root@localhost:8086/testdb", n_retries=2
        )
        forwarder.dataframe_client = mock_influx_dataframe_client
        forwarder._write_to_influx_with_retries(df, "start")
        find_caplog_record(
            "WARNING",
            "Failed to forward data to influx on attempt 1 out of 2.\nError: Connection refused.\nSleeping 8 seconds and trying again.",
        )
        find_caplog_record(
            "WARNING",
            "Failed to forward data to influx on attempt 2 out of 2.\nError: Connection refused.\nSleeping 16 seconds and trying again.",
        )
        find_caplog_record(
            "ERROR", "Failed to forward data to influx. Error: Connection refused"
        )


def get_test_data(columns: List[str]) -> pd.DataFrame:
    """
    Generate timeseries sensor dataframe with given columns
    """
    index = pd.date_range("2019-01-01", "2019-01-02", periods=4)
    df = pd.DataFrame(columns=columns, index=index)

    # Generate some unique values for each key, and insert it into that column
    for i, key in enumerate(columns):
        df[key] = range(i, i + 4)
    return df


def test_influx_forwarder(influxdb, influxdb_uri, sensors, sensors_str):
    """
    Test that the forwarder creates correct points from a
    multi-indexed series
    """
    with patch.object(sensor_tag, "_asset_from_tag_name", return_value="default"):
        machine = Machine.from_config(
            config={
                "name": "some-target-name",
                "dataset": {
                    "tags": sensors_str,
                    "target_tag_list": sensors_str,
                    "train_start_date": "2016-01-01T00:00:00Z",
                    "train_end_date": "2016-01-05T00:00:00Z",
                    "resolution": "10T",
                },
                "model": "sklearn.linear_model.LinearRegression",
            },
            project_name="test-project",
        )

    # Feature outs which match length of tags
    # These should then be re-mapped to the sensor tag names
    input_keys = [("name1", i) for i, _ in enumerate(sensors)]

    # Feature outs which don't match the length of the tags
    # These will be kept at 0..N as field names
    # output_keys = [("name2", f"sensor_{i}") for i in range(len(sensors) * 2)]
    output_keys = [("name2", i) for i in range(len(sensors) * 2)]

    # Assign all keys unique numbers
    df = get_test_data(pd.MultiIndex.from_tuples(input_keys + output_keys))

    # Create the forwarder and forward the 'predictions' to influx.
    forwarder = ForwardPredictionsIntoInflux(destination_influx_uri=influxdb_uri)
    forwarder.forward_predictions(predictions=df, machine=machine)

    # Client to manually verify the points written
    client = influx_client_from_uri(influxdb_uri, dataframe_client=True)

    name1_results = client.query("SELECT * FROM name1")["name1"]

    # Should have column names: 'machine', 'sensor_name', 'sensor_value'
    assert all(
        c in name1_results.columns for c in ["machine", "sensor_name", "sensor_value"]
    )

    # Check that values returned from InfluxDB match what put in for inputs
    for i, tag in enumerate(sensors_str):
        results_mask = name1_results["sensor_name"] == tag
        assert np.allclose(
            df[("name1", i)].values, name1_results[results_mask]["sensor_value"].values
        )

    # Now check the other top level name "name2" is a measurement with the correct points written
    name2_results = client.query("SELECT * FROM name2")["name2"]

    # Should have the same names as tags, since all top levels get stacked into the same resulting columns
    assert all(
        [c in name2_results.columns for c in ["machine", "sensor_name", "sensor_value"]]
    )

    # Check that values returned from InfluxDB match what put in for outputs
    # Note that here the influx sensor names for the output tags are string-cast integers
    for key in output_keys:
        results_mask = name2_results["sensor_name"] == str(key[1])
        assert np.allclose(
            df[key].values, name2_results[results_mask]["sensor_value"].values
        )


def test_influx_send_data(influxdb, influxdb_uri, sensors, sensors_str):
    """
    """
    df = get_test_data(sensors_str)

    # Create the forwarder and forward the sensor data to influx.
    forwarder = ForwardPredictionsIntoInflux(destination_influx_uri=influxdb_uri)
    forwarder.send_sensor_data(df)

    # Client to manually verify the points written
    client = influx_client_from_uri(influxdb_uri, dataframe_client=True)
    resampled_results = client.query("SELECT * FROM resampled")["resampled"]

    # Should have column names: 'sensor_name', 'sensor_value'
    assert all(c in resampled_results.columns for c in ["sensor_name", "sensor_value"])

    # Check that values returned from InfluxDB match what put in for inputs
    for key in sensors_str:
        results_mask = resampled_results["sensor_name"] == key
        assert np.allclose(
            df[key].values, resampled_results[results_mask]["sensor_value"].values
        )
