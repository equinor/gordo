# -*- coding: utf-8 -*-

import io
import logging

import pytest
import pandas as pd
import numpy as np

from gordo.server import utils as server_utils
from gordo.server.views.base import BaseModelView
from gordo_dataset.sensor_tag import SensorTag
from flask import Flask, g


def test_empty_target_tag_list():
    app = Flask(__name__)
    test_tag = SensorTag("test", "asset")
    with app.app_context():
        g.metadata = {"dataset": {"tag_list": [test_tag]}}
        view = BaseModelView()
        assert view.target_tags == [test_tag]


@pytest.mark.parametrize(
    "data_size,to_dict_arg",
    [(10, None), (1, None), (10, "records"), (10, "list"), (10, "dict")],
)
@pytest.mark.parametrize("resp_format", ("json", "parquet", None))
@pytest.mark.parametrize("send_as_parquet", (True, False))
def test_prediction_endpoint_post_ok(
    base_route,
    sensors,
    sensors_str,
    gordo_ml_server_client,
    data_size,
    to_dict_arg,
    resp_format,
    send_as_parquet,
):
    """
    Test the expected successful data posts, by sending a variety of valid
    JSON formats of a dataframe, as well as parquet serializations.
    """
    data_to_post = np.random.random(size=(data_size, len(sensors))).tolist()

    if to_dict_arg is not None:
        df = pd.DataFrame(data_to_post, columns=sensors_str)
        data_to_post = df.to_dict(to_dict_arg)

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


def test_prediction_endpoint_post_fail(
    caplog, base_route, sensors, gordo_ml_server_client
):
    """
    Test expected failures when posting certain types of data
    """
    erroneous_datasets = [
        # Not enough features
        {"X": [list(range(len(sensors) - 1))]},
        # No X
        {"no-X-here": True},
        # Badly formatted data
        {"X": [[1, 2, [1, 2]]]},
    ]
    for i, data_to_post in enumerate(erroneous_datasets):
        with caplog.at_level(logging.CRITICAL):
            resp = gordo_ml_server_client.post(
                f"{base_route}/prediction", json=data_to_post
            )
        assert (
            resp.status_code == 400
        ), f"Prediction did not fail with data, {str(data_to_post)}."
