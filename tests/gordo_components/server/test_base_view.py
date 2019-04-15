# -*- coding: utf-8 -*-

import logging

import pytest
import numpy as np

import tests.utils as tu


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

    assert "data" in data
    np.asanyarray(data["data"])  # And should be able to cast into array


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
