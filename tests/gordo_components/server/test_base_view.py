# -*- coding: utf-8 -*-

import logging

import pytest
import numpy as np
import pandas as pd

from gordo_components.server.views.base import BaseModelView
from gordo_components.dataset.sensor_tag import normalize_sensor_tags

import tests.utils as tu


@pytest.mark.parametrize(
    "dates", [pd.date_range("2016-01-01", "2016-01-02", periods=10), None]
)
@pytest.mark.parametrize("tags", [["tag1", "tag2"], ["tag"]])
@pytest.mark.parametrize("output_features_equals_input", (True, False))
@pytest.mark.parametrize("output_offset", (0, 1, 2, 3))
def test_base_dataframe_creation(
    tags, dates, output_features_equals_input, output_offset
):

    names = (
        "original-input",
        "model-output",
        "transformed-model-input",
        "inverse-transformed-model-output",
    )

    # Make some fake data base on dates and tags
    size = len(dates if dates is not None else list(range(10))) * len(tags)
    values = {
        name: np.random.random(size=size).reshape(-1, len(tags)) for name in names
    }

    # Simulate where the model's output may not match the shape of it's input
    if not output_features_equals_input:
        values["model-output"] = np.random.random(
            len(values["original-input"])
        ).reshape(-1, 1)
        values.pop("inverse-transformed-model-output")

    # simulate where model's output length doesn't match it's input length
    # ie. as with an LSTM which outputs the offset of it's lookback window
    values["model-output"] = values["model-output"][output_offset:]

    # If we have inversed model output, ensure that is also the same length as model output
    if "inverse-transformed-model-output" in values:
        values["inverse-transformed-model-output"] = values[
            "inverse-transformed-model-output"
        ][output_offset:]

    # pass in the arrays, of which model output's may be different lengths / shapes than the input
    # but should provide a valid dataframe in all cases.
    df = BaseModelView.make_base_dataframe(
        tags=tags,
        original_input=values["original-input"],
        model_output=values["model-output"],
        transformed_model_input=values["transformed-model-input"],
        inverse_transformed_model_output=values.get("inverse-transformed-model-output"),
        index=dates,
    )

    # Loop over each top level column name and verify the values in the dataframe
    for column in names:

        # If model output does not equal input, we cannot have the inverse transformed of it
        if (
            not output_features_equals_input
            and column == "inverse-transformed-model-output"
        ):
            assert column not in df.columns

        # Otherwise, ensure the assigned values under that top-level column name
        # matches the values passed to the function
        else:
            # offset column's like 'original-input' since it will be offsetted inside make_base_dataframe()
            assert np.array_equal(df[column].values, values[column][-len(df) :])

    # Test expected index if dates were supplied or not
    if dates is not None:
        assert np.array_equal(df.index.values, dates.values[output_offset:])
    else:
        assert np.array_equal(df.index.values, np.arange(0, len(df)))


@pytest.mark.parametrize(
    "data_to_post",
    [
        {"X": np.random.random(size=(10, len(tu.SENSORS_STR_LIST))).tolist()},
        {"X": np.random.random(size=len(tu.SENSORS_STR_LIST)).tolist()},
    ],
)
def test_prediction_endpoint_post_ok(sensors, gordo_ml_server_client, data_to_post):
    """
    Test the expected successfull data posts
    """
    resp = gordo_ml_server_client.post("/prediction", json=data_to_post)
    assert resp.status_code == 200

    data = resp.get_json()

    # Data should be a list of dicts
    assert "data" in data
    assert isinstance(data["data"], list)

    # One record (dict) should have keys mapped to lists of values
    record = data["data"][0]
    for key in (
        "inverse-transformed-model-output",
        "model-output",
        "original-input",
        "transformed-model-input",
    ):
        assert key in record
        assert isinstance(record[key], list)


@pytest.mark.parametrize(
    "data_to_post",
    [{"X": [[1, 2, 3], [1, 2, 3]]}, {"no-x-here": True}, {"X": [[1, 2, 3], [1, 2]]}],
)
def test_prediction_endpoint_post_fail(
    caplog, sensors, gordo_ml_server_client, data_to_post
):
    """
    Test expected failures when posting certain types of data
    """
    with caplog.at_level(logging.CRITICAL):
        resp = gordo_ml_server_client.post("/prediction", json=data_to_post)
    assert resp.status_code == 400
