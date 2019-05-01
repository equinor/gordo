# -*- coding: utf-8 -*-

import logging

import pytest
import numpy as np

from gordo_components import serializer
from tests import utils as tu


logger = logging.getLogger(__name__)


def test_healthcheck_endpoint(gordo_ml_server_client):
    """
    Test expected behavior of /healthcheck
    """
    resp = gordo_ml_server_client.get("/healthcheck")
    assert resp.status_code == 200

    data = resp.get_json()
    logger.debug(f"Got resulting JSON response: {data}")
    assert "gordo-server-version" in data


def test_metadata_endpoint(gordo_ml_server_client):
    """
    Test the expected behavior of /metadata
    """
    resp = gordo_ml_server_client.get("/metadata")
    assert resp.status_code == 200

    data = resp.get_json()
    assert "metadata" in data
    assert data["metadata"]["user-defined"]["model-name"] == "test-model"


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


@pytest.mark.parametrize(
    "data_to_post",
    [
        {"X": np.random.random(size=(10, len(tu.SENSORTAG_LIST))).tolist()},
        {"X": np.random.random(size=len(tu.SENSORTAG_LIST)).tolist()},
    ],
)
def test_prediction_endpoint_post_ok(sensors, gordo_ml_server_client, data_to_post):
    """
    Test the expected successfull data posts
    """
    resp = gordo_ml_server_client.post("/prediction", json=data_to_post)
    assert resp.status_code == 200

    data = resp.get_json()

    assert "output" in data
    np.asanyarray(data["output"])  # And should be able to cast into array


def test_download_model(gordo_ml_server_client):
    """
    Test we can download a model, loadable via serializer.loads()
    """
    resp = gordo_ml_server_client.get("/download-model")

    serialized_model = resp.get_data()
    model = serializer.loads(serialized_model)

    # All models have a fit method
    assert hasattr(model, "fit")

    # Models MUST have either predict or transform
    assert hasattr(model, "predict") or hasattr(model, "transform")


def test_prediction_endpoint_get(influxdb, gordo_ml_server_client):
    """
    Client can ask for a timerange based prediction
    """
    resp = gordo_ml_server_client.get(
        "/prediction",
        json={"start": "2016-01-01T00:00:00+00:00", "end": "2016-01-01T12:00:00+00:00"},
    )

    assert resp.status_code == 200
    assert "output" in resp.json

    # Verify keys & structure of one output record output contains a list of:
    # {'start': timestamp, 'end': timestamp, 'tags': {'tag': float}, 'total_abnormality': float}
    assert "start" in resp.json["output"][0]
    assert "end" in resp.json["output"][0]
    assert "tags" in resp.json["output"][0]
    assert isinstance(resp.json["output"][0]["tags"], dict)
    assert "total_anomaly" in resp.json["output"][0]

    # Request greater than 1 day should be a bad request
    resp = gordo_ml_server_client.get(
        "/prediction",
        json={"start": "2016-01-01T00:00:00+00:00", "end": "2016-01-03T12:00:00+00:00"},
    )
    assert resp.status_code == 400
    assert "error" in resp.json

    # Requests for overlapping time sample buckets should only produce
    # predictions whose bucket falls completely within the requested
    # time range

    # This should give one prediction with start end of
    # 2016-01-01 00:10:00+00:00 and 2016-01-01 00:20:00+00:00, respectively.
    # because it will not go over, ie. giving a bucket from 00:20:00 to 00:30:00
    # but can give a bucket which contains data before the requested start date.
    resp = gordo_ml_server_client.get(
        "/prediction",
        json={"start": "2016-01-01T00:11:00+00:00", "end": "2016-01-01T00:21:00+00:00"},
    )
    assert resp.status_code == 200
    assert len(resp.json["output"]) == 1, f"Expected one prediction, got: {resp.json}"
    assert resp.json["output"][0]["start"] == "2016-01-01 00:10:00+00:00"
    assert resp.json["output"][0]["end"] == "2016-01-01 00:20:00+00:00"
