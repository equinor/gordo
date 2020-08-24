import pytest
from flask import Flask
from mock import patch

from gordo.server.prometheus import GordoServerPrometheusMetrics
from prometheus_client import CollectorRegistry


@pytest.fixture
def registry():
    return CollectorRegistry()


@pytest.fixture
def prometheus_metrics(registry):
    return GordoServerPrometheusMetrics(
        args_labels=(("gordo_project", "project"), ("gordo_name", "model")),
        info={"version": "0.60.0"},
        ignore_paths=["/healthcheck"],
        registry=registry,
    )


@pytest.fixture
def gordo_server(prometheus_metrics: GordoServerPrometheusMetrics):
    app = Flask("test_server")

    @app.route("/healthcheck")
    def healthcheck():
        return "", 200

    @app.route("/success/<gordo_project>/<gordo_name>")
    def success(gordo_project: str, gordo_name: str):
        return "", 200

    @app.route("/failed/<gordo_project>")
    def failed(gordo_project: str):
        return "", 500

    with patch("gordo.server.prometheus.current_time", side_effect=[0.0, 0.25]):
        prometheus_metrics.prepare_app(app)

        yield app


def test_success(gordo_server: Flask, registry: CollectorRegistry):
    client = gordo_server.test_client()
    client.get("/success/project1/gordo1")
    sample_value = registry.get_sample_value("gordo_server_info", {"version": "0.60.0"})
    assert sample_value == 1.0, "Metric gordo_server_info != 1.0"
    sample_value = registry.get_sample_value(
        "gordo_server_request_duration_seconds_bucket",
        {
            "version": "0.60.0",
            "project": "project1",
            "model": "gordo1",
            "method": "GET",
            "path": "/success/<gordo_project>/<gordo_name>",
            "status_code": "200",
            "le": "0.25",
        },
    )
    assert (
        sample_value == 1.0
    ), "Metric gordo_server_request_duration_seconds_bucket != 1.0"
    sample_value = registry.get_sample_value(
        "gordo_server_requests_total",
        {
            "version": "0.60.0",
            "project": "project1",
            "model": "gordo1",
            "method": "GET",
            "path": "/success/<gordo_project>/<gordo_name>",
            "status_code": "200",
        },
    )
    assert sample_value == 1.0, "Metric gordo_server_requests_total != 1.0"


def test_failed(gordo_server: Flask, registry: CollectorRegistry):
    client = gordo_server.test_client()
    client.get("/failed/project1")
    sample_value = registry.get_sample_value(
        "gordo_server_request_duration_seconds_bucket",
        {
            "version": "0.60.0",
            "project": "project1",
            "model": "",
            "method": "GET",
            "path": "/failed/<gordo_project>",
            "status_code": "500",
            "le": "0.25",
        },
    )
    assert (
        sample_value == 1.0
    ), "Metric gordo_server_request_duration_seconds_bucket != 1.0"
    sample_value = registry.get_sample_value(
        "gordo_server_requests_total",
        {
            "version": "0.60.0",
            "project": "project1",
            "model": "",
            "method": "GET",
            "path": "/failed/<gordo_project>",
            "status_code": "500",
        },
    )
    assert sample_value == 1.0, "Metric gordo_server_requests_total != 1.0"


def test_ignore(gordo_server: Flask, registry: CollectorRegistry):
    client = gordo_server.test_client()
    client.get("/healthcheck")
    sample_value = registry.get_sample_value(
        "gordo_server_requests_total",
        {
            "version": "0.60.0",
            "project": "",
            "model": "",
            "method": "GET",
            "path": "/healthcheck",
            "status_code": "200",
        },
    )
    assert sample_value is None, "Metric gordo_server_requests_total is not None"
