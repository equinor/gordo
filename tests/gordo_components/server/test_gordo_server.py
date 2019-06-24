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


def test_response_header_timing(gordo_ml_server_client):
    """
    Test that the response contains a `Server-Timing` header
    """
    resp = gordo_ml_server_client.get("/healthcheck")
    assert resp.status_code == 200
    assert "Server-Timing" in resp.headers
    assert "request_walltime_s" in resp.headers["Server-Timing"]


def test_metadata_endpoint(gordo_ml_server_client):
    """
    Test the expected behavior of /metadata
    """
    resp = gordo_ml_server_client.get("/metadata")
    assert resp.status_code == 200

    data = resp.get_json()
    assert "metadata" in data
    assert data["metadata"]["user-defined"]["model-name"] == "test-model"


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
