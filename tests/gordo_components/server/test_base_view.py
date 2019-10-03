# -*- coding: utf-8 -*-

import io
import logging

import pytest
import pandas as pd
import numpy as np

from gordo_components.server import utils as server_utils
import tests.utils as tu


@pytest.mark.parametrize(
    "data_to_post",
    [
        np.random.random(size=(10, len(tu.SENSORS_STR_LIST))).tolist(),
        np.random.random(size=(1, len(tu.SENSORS_STR_LIST))).tolist(),
        pd.DataFrame(
            np.random.random((10, len(tu.SENSORS_STR_LIST))),
            columns=tu.SENSORS_STR_LIST,
        ).to_dict("records"),
        pd.DataFrame(
            np.random.random((10, len(tu.SENSORS_STR_LIST))),
            columns=tu.SENSORS_STR_LIST,
        ).to_dict("list"),
        pd.DataFrame(
            np.random.random((10, len(tu.SENSORS_STR_LIST))),
            columns=tu.SENSORS_STR_LIST,
        ).to_dict("dict"),
    ],
)
@pytest.mark.parametrize("resp_format", ("json", "parquet", None))
@pytest.mark.parametrize("send_as_parquet", (True, False))
def test_prediction_endpoint_post_ok(
    base_route,
    sensors,
    gordo_ml_server_client,
    data_to_post,
    resp_format,
    send_as_parquet,
):
    """
    Test the expected successful data posts, by sending a variety of valid
    JSON formats of a dataframe, as well as parquet serializations.
    """
    endpoint = f"{base_route}/prediction"
    if resp_format is not None:
        endpoint += f"?format={resp_format}"

    if send_as_parquet:
        X = pd.DataFrame.from_dict(data_to_post)
        kwargs = dict(
            data={"X": (io.BytesIO(server_utils.dataframe_into_parquet_bytes(X)), "X")}
        )
    else:
        kwargs = dict(json={"X": data_to_post})

    resp = gordo_ml_server_client.post(endpoint, **kwargs)
    assert resp.status_code == 200

    if resp_format in (None, "json"):
        data = server_utils.dataframe_from_dict(resp.json["data"])
    else:
        data = server_utils.dataframe_from_parquet_bytes(resp.data)

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
    caplog, base_route, sensors, gordo_ml_server_client, data_to_post
):
    """
    Test expected failures when posting certain types of data
    """
    with caplog.at_level(logging.CRITICAL):
        resp = gordo_ml_server_client.post(
            f"{base_route}/prediction", json=data_to_post
        )
    assert resp.status_code == 400
