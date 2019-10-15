# -*- coding: utf-8 -*-

import logging

import pytest
import subprocess

from gordo_components.server.server import run_cmd
from gordo_components import serializer


logger = logging.getLogger(__name__)


def test_healthcheck_endpoint(base_route, gordo_ml_server_client):
    """
    Test expected behavior of /<gordo-name>/healthcheck
    """
    # Should also be at the very lowest level as well.
    resp = gordo_ml_server_client.get(f"/healthcheck")
    assert resp.status_code == 200

    resp = gordo_ml_server_client.get(f"{base_route}/healthcheck")
    assert resp.status_code == 200

    data = resp.get_json()
    logger.debug(f"Got resulting JSON response: {data}")
    assert "gordo-server-version" in data


def test_response_header_timing(base_route, gordo_ml_server_client):
    """
    Test that the response contains a `Server-Timing` header
    """
    resp = gordo_ml_server_client.get(f"{base_route}/healthcheck")
    assert resp.status_code == 200
    assert "Server-Timing" in resp.headers
    assert "request_walltime_s" in resp.headers["Server-Timing"]


def test_metadata_endpoint(base_route, gordo_ml_server_client):
    """
    Test the expected behavior of /metadata
    """
    resp = gordo_ml_server_client.get(f"{base_route}/metadata")
    assert resp.status_code == 200

    data = resp.get_json()
    assert "metadata" in data
    assert data["metadata"]["user-defined"]["model-name"] == "test-model"


def test_download_model(base_route, gordo_ml_server_client):
    """
    Test we can download a model, loadable via serializer.loads()
    """
    resp = gordo_ml_server_client.get(f"{base_route}/download-model")

    serialized_model = resp.get_data()
    model = serializer.loads(serialized_model)

    # All models have a fit method
    assert hasattr(model, "fit")

    # Models MUST have either predict or transform
    assert hasattr(model, "predict") or hasattr(model, "transform")


def test_run_cmd(monkeypatch):
    """
    Test that execution error catchings work as expected
    """

    # Call command that raises FileNotFoundError, a subclass of OSError
    cmd = ["gumikorn", "gordo_components.server.server:app"]
    with pytest.raises(OSError):
        run_cmd(cmd)

    # Call command that raises a CalledProcessError
    cmd = ["ping", "--bad-option"]
    with pytest.raises(subprocess.CalledProcessError):
        run_cmd(cmd)
