# -*- coding: utf-8 -*-

import logging

import pytest
import pandas as pd
import numpy as np

from gordo_components.server import utils as server_utils
import tests.utils as tu


@pytest.mark.parametrize(
    "data_to_post",
    [
        {"X": np.random.random(size=(10, len(tu.SENSORS_STR_LIST))).tolist()},
        {"X": np.random.random(size=(1, len(tu.SENSORS_STR_LIST))).tolist()},
        {
            "X": pd.DataFrame(
                np.random.random((10, len(tu.SENSORS_STR_LIST))),
                columns=tu.SENSORS_STR_LIST,
            ).to_dict("records")
        },
        {
            "X": pd.DataFrame(
                np.random.random((10, len(tu.SENSORS_STR_LIST))),
                columns=tu.SENSORS_STR_LIST,
            ).to_dict("list")
        },
        {
            "X": pd.DataFrame(
                np.random.random((10, len(tu.SENSORS_STR_LIST))),
                columns=tu.SENSORS_STR_LIST,
            ).to_dict("dict")
        },
    ],
)
def test_prediction_endpoint_post_ok(sensors, gordo_ml_server_client, data_to_post):
    """
    Test the expected successfull data posts
    """
    resp = gordo_ml_server_client.post("/prediction", json=data_to_post)

    assert resp.status_code == 200

    data = server_utils.dataframe_from_dict(resp.json["data"])

    # Expected column names
    assert all(key in data for key in ("model-output", "model-input"))


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
