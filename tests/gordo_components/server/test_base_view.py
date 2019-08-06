# -*- coding: utf-8 -*-

import logging

import pytest
import numpy as np

import tests.utils as tu


@pytest.mark.parametrize(
    "data_to_post",
    [
        {"X": np.random.random(size=(10, len(tu.SENSORS_STR_LIST))).tolist()},
        {"X": np.random.random(size=(1, len(tu.SENSORS_STR_LIST))).tolist()},
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
    for key in ("model-output", "model-input"):
        assert key in record
        assert isinstance(record[key], list)


@pytest.mark.parametrize(
    "data_to_post",
    [
        {"X": [list(range(len(tu.SENSORTAG_LIST) - 1))]},  # Not enough features
        {"no-x-here": True},  # No X
        {"X": [[1, 2, [1, 2]]]},  # Badly formatted data
    ],
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
