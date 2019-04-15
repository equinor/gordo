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

    assert "data" in data
    np.asanyarray(data["data"])  # And should be able to cast into array


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
