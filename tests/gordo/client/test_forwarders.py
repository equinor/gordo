# -*- coding: utf-8 -*-

from unittest.mock import patch

import pandas as pd
import numpy as np

from gordo.client.forwarders import ForwardPredictionsIntoInflux
from gordo.client.utils import influx_client_from_uri
from gordo.machine import Machine
from gordo.machine.dataset import sensor_tag


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
    keys = [("name1", i) for i, _ in enumerate(sensors)]

    # Feature outs which don't match the length of the tags
    # These will be kept at 0..N as field names
    keys.extend([("name2", i) for i in range(len(sensors) * 2)])

    # Assign all keys unique numbers
    columns = pd.MultiIndex.from_tuples(keys)
    index = pd.date_range("2019-01-01", "2019-01-02", periods=4)
    df = pd.DataFrame(columns=columns, index=index)

    # Generate some unique values for each key, and insert it into that column
    for i, key in enumerate(keys):
        df[key] = range(i, i + 4)

    # Create the forwarder and forward the 'predictions' to influx.
    forwarder = ForwardPredictionsIntoInflux(destination_influx_uri=influxdb_uri)
    forwarder.forward_predictions(predictions=df, machine=machine)

    # Client to manually verify the points written
    client = influx_client_from_uri(influxdb_uri, dataframe_client=True)

    name1_results = client.query("SELECT * FROM name1")["name1"]

    # Should have the tag names as column names since the shape matched
    assert all(c in name1_results.columns for c in ["machine"] + sensors_str)
    for i, tag in enumerate(sensors_str):
        assert np.allclose(df[("name1", i)].values, name1_results[tag].values)

    # Now check the other top level name "name2" is a measurement with the correct points written
    name2_results = client.query("SELECT * FROM name2")["name2"]

    # Should not have the same names as tags, since shape was 2x as long, should just be numeric columns
    assert all(
        [
            str(c) in name2_results.columns
            for c in ["machine"] + list(range(len(sensors) * 2))
        ]
    )
    for key in filter(lambda k: k[0] == "name2", keys):
        assert np.allclose(df[key].values, name2_results[str(key[1])].values)
