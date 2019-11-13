# -*- coding: utf-8 -*-

import os
import json
import logging
import tempfile
import typing
from dateutil.parser import isoparse  # type: ignore

import pytest
import requests
import pandas as pd
import numpy as np
from unittest import mock
from click.testing import CliRunner
from sklearn.base import BaseEstimator
from mock import patch, call

from gordo_components.client import Client, utils as client_utils
from gordo_components.client.utils import EndpointMetadata
from gordo_components.client.io import (
    _handle_response,
    HttpUnprocessableEntity,
    BadRequest,
)
from gordo_components.client.forwarders import ForwardPredictionsIntoInflux
from gordo_components.client.utils import PredictionResult
from gordo_components.data_provider import providers
from gordo_components.server import utils as server_utils
from gordo_components.model import utils as model_utils
from gordo_components import cli, serializer
from gordo_components.cli import custom_types

from tests import utils as tu


def test_client_get_metadata(watchman_service):
    """
    Test client's ability to get metadata from some target
    """
    client = Client(project=tu.GORDO_PROJECT)

    metadata = client.get_metadata()
    assert isinstance(metadata, dict)

    # Can't get metadata for non-existent target
    with pytest.raises(ValueError):
        client = Client(project=tu.GORDO_PROJECT, target="no-such-target")
        client.get_metadata()


def test_client_predict_specific_targets(watchman_service):
    """
    Client.predict should filter any endpoints given to it.
    """
    client = Client(project=tu.GORDO_PROJECT)
    with mock.patch.object(
        Client,
        "predict_single_endpoint",
        return_value=PredictionResult("test-name", [], []),
    ) as patched:

        start = (isoparse("2016-01-01T00:00:00+00:00"),)
        end = isoparse("2016-01-01T12:00:00+00:00")

        # Should not actually call any predictions because this machine name doesn't exist
        client.predict(start=start, end=end, endpoint_names=["non-existant-machine"])
        patched.assert_not_called()

        # Should be called once, for this machine.
        client.predict(start=start, end=end, endpoint_names=[tu.GORDO_SINGLE_TARGET])
        patched.assert_called_once()


def test_client_download_model(watchman_service):
    """
    Test client's ability to download the model
    """
    client = Client(project=tu.GORDO_PROJECT, target=tu.GORDO_SINGLE_TARGET)

    models = client.download_model()
    assert isinstance(models, dict)
    assert isinstance(models[tu.GORDO_SINGLE_TARGET], BaseEstimator)

    # Can't download model for non-existent target
    with pytest.raises(ValueError):
        client = Client(project=tu.GORDO_PROJECT, target="non-existent-target")
        client.download_model()


@pytest.mark.parametrize("batch_size", (10, 100))
@pytest.mark.parametrize("use_parquet", (True, False))
def test_client_predictions_diff_batch_sizes(
    influxdb, watchman_service, batch_size: int, use_parquet: bool
):
    """
    Run the prediction client with different batch-sizes and whether to use
    a data provider or not.
    """
    # Time range used in this test
    start, end = (
        isoparse("2016-01-01T00:00:00+00:00"),
        isoparse("2016-01-01T12:00:00+00:00"),
    )

    # Client only used within the this test
    test_client = client_utils.influx_client_from_uri(tu.INFLUXDB_URI)

    # Created measurements by prediction client with dest influx
    query = f"""
    SELECT *
    FROM "model-output"
    WHERE("machine" =~ /^{tu.GORDO_SINGLE_TARGET}$/)
    """

    # Before predicting, influx destination db should be empty for 'predictions' measurement
    vals = test_client.query(query)
    assert len(vals) == 0

    data_provider = providers.InfluxDataProvider(
        measurement=tu.INFLUXDB_MEASUREMENT,
        value_name="Value",
        client=client_utils.influx_client_from_uri(
            uri=tu.INFLUXDB_URI, dataframe_client=True
        ),
    )

    prediction_client = Client(
        project=tu.GORDO_PROJECT,
        data_provider=data_provider,
        prediction_forwarder=ForwardPredictionsIntoInflux(  # type: ignore
            destination_influx_uri=tu.INFLUXDB_URI
        ),
        batch_size=batch_size,
        use_parquet=use_parquet,
        parallelism=10,
    )

    # Should have discovered machine-1
    assert len(prediction_client.endpoints) == 1

    # All endpoints should be healthy
    assert all(ep.healthy for ep in prediction_client.endpoints)

    # Get predictions
    predictions = prediction_client.predict(start=start, end=end)
    assert isinstance(predictions, list)
    assert len(predictions) == 1

    name, predictions, error_messages = predictions[0]  # First dict of predictions
    assert isinstance(name, str)
    assert isinstance(predictions, pd.DataFrame)
    assert isinstance(error_messages, list)

    assert isinstance(predictions.index, pd.core.indexes.datetimes.DatetimeIndex)

    # This should have resulted in writting predictions to influx
    # Before predicting, influx destination db should be empty
    vals = test_client.query(query)
    assert (
        len(vals) > 0
    ), f"Expected new values in 'predictions' measurement, but found {vals}"


@pytest.mark.parametrize(
    "args",
    [
        ["client", "--help"],
        ["client", "predict", "--help"],
        ["client", "metadata", "--help"],
        ["client", "download-model", "--help"],
    ],
)
def test_client_cli_basic(args):
    """
    Test that client specific subcommands exist
    """
    runner = CliRunner()
    out = runner.invoke(cli.gordo, args=args)
    assert (
        out.exit_code == 0
    ), f"Expected output code 0 got '{out.exit_code}', {out.output}"


def test_client_cli_metadata(watchman_service, tmp_dir):
    """
    Test proper execution of client predict sub-command
    """
    runner = CliRunner()

    # Simple metadata fetching
    out = runner.invoke(
        cli.gordo,
        args=[
            "client",
            "--project",
            tu.GORDO_PROJECT,
            "--target",
            tu.GORDO_SINGLE_TARGET,
            "metadata",
        ],
    )
    assert out.exit_code == 0
    assert tu.GORDO_SINGLE_TARGET in out.output

    # Save metadata to file
    output_file = os.path.join(tmp_dir, "metadata.json")
    out = runner.invoke(
        cli.gordo,
        args=[
            "client",
            "--project",
            tu.GORDO_PROJECT,
            "--target",
            tu.GORDO_SINGLE_TARGET,
            "metadata",
            "--output-file",
            output_file,
        ],
    )
    assert out.exit_code == 0, f"{out.exc_info}"
    assert os.path.exists(output_file)
    with open(output_file) as f:
        metadata = json.load(f)
        assert tu.GORDO_SINGLE_TARGET in metadata


def test_client_cli_download_model(watchman_service, tmp_dir):
    """
    Test proper execution of client predict sub-command
    """
    runner = CliRunner()

    # Empty output directory before downloading
    assert len(os.listdir(tmp_dir)) == 0

    out = runner.invoke(
        cli.gordo,
        args=[
            "client",
            "--project",
            tu.GORDO_PROJECT,
            "--target",
            tu.GORDO_SINGLE_TARGET,
            "download-model",
            tmp_dir,
        ],
    )
    assert (
        out.exit_code == 0
    ), f"Expected output code 0 got '{out.exit_code}', {out.output}"

    # Output directory should not be empty any longer
    assert len(os.listdir(tmp_dir)) > 0

    model_output_dir = os.path.join(tmp_dir, tu.GORDO_SINGLE_TARGET)
    assert os.path.isdir(model_output_dir)

    model = serializer.load(model_output_dir)
    assert isinstance(model, BaseEstimator)


@pytest.mark.parametrize(
    "forwarder_args",
    [["--influx-uri", tu.INFLUXDB_URI, "--forward-resampled-sensors"], None],
)
@pytest.mark.parametrize("output_dir", [True, False])
@pytest.mark.parametrize("use_parquet", (True, False))
@pytest.mark.parametrize("session_config", ({}, {"headers": {}}))
def test_client_cli_predict(
    influxdb,
    gordo_name,
    watchman_service,
    tmp_dir,
    forwarder_args,
    trained_model_directory,
    output_dir,
    use_parquet,
    session_config,
):
    """
    Test ability for client to get predictions via CLI
    """
    runner = CliRunner()

    args = ["client", "--metadata", "key,value", "--project", tu.GORDO_PROJECT]
    if session_config:
        args.extend(["--session-config", json.dumps(session_config)])

    args.extend(
        [
            "predict",
            "--parquet" if use_parquet else "--no-parquet",
            "2016-01-01T00:00:00Z",
            "2016-01-01T01:00:00Z",
        ]
    )

    influx_client = client_utils.influx_client_from_uri(
        uri=tu.INFLUXDB_URI, dataframe_client=True
    )
    query = """
        SELECT *
        FROM "resampled"
        """

    # Do we have forwarder args?
    if forwarder_args is not None:
        args.extend(forwarder_args)
        vals = influx_client.query(query)
        # There is no data there before we start doing things
        assert len(vals) == 0

    # Should it write out the predictions to dataframes in an output directory?
    if output_dir:
        args.extend(["--output-dir", tmp_dir])

    # Do we have a data provider, POST else GET requests
    args.extend(
        ["--data-provider", json.dumps(providers.RandomDataProvider().to_dict())]
    )

    # Run without any error
    out = runner.invoke(cli.gordo, args=args)
    assert out.exit_code == 0, f"{out.output}"

    # If we activated forwarder and we had any actual data then there should
    # be resampled values in the influx
    if forwarder_args:
        vals = influx_client.query(query)
        assert len(vals) == 1
        assert len(vals["resampled"]) == 48
        influx_client.drop_measurement("resampled")

    # Did it save dataframes to output dir if specified?
    if output_dir:
        assert os.path.exists(os.path.join(tmp_dir, f"{tu.GORDO_SINGLE_TARGET}.csv.gz"))


@pytest.mark.parametrize(
    "should_fail,start_date,end_date",
    [
        (True, "1888-01-01T00:00:00Z", "1888-02-01T01:00:00Z"),  # Fail on bad dates
        (False, "2016-01-01T00:00:00Z", "2016-01-01T01:00:00Z"),  # pass on good dates
    ],
)
def test_client_cli_predict_non_zero_exit(
    should_fail, start_date, end_date, caplog, influxdb, watchman_service
):
    """
    Test ability for client to get predictions via CLI
    """
    runner = CliRunner()

    # Should fail requesting dates which clearly don't exist.
    args = [
        "client",
        "--metadata",
        "key,value",
        "--project",
        tu.GORDO_PROJECT,
        "predict",
        start_date,
        end_date,
    ]

    data_provider = providers.InfluxDataProvider(
        measurement=tu.INFLUXDB_MEASUREMENT, value_name="Value", uri=tu.INFLUXDB_URI
    )

    args.extend(["--data-provider", json.dumps(data_provider.to_dict())])

    # Run without any error
    with caplog.at_level(logging.CRITICAL):
        out = runner.invoke(cli.gordo, args=args)

    if should_fail:
        assert out.exit_code != 0, f"{out.output or out.exception}"
    else:
        assert out.exit_code == 0, f"{out.output or out.exception}"


@pytest.mark.parametrize(
    "config",
    (
        '{"type": "RandomDataProvider", "max_size": 200}',
        '{"type": "InfluxDataProvider", "measurement": "value"}',
    ),
)
def test_data_provider_click_param(config):
    """
    Test click custom param to load a provider from a string config representation
    """
    expected_provider_type = json.loads(config)["type"]
    provider = custom_types.DataProviderParam()(config)
    assert isinstance(provider, getattr(providers, expected_provider_type))

    # Should also be able to take a file path with the json
    with tempfile.NamedTemporaryFile(mode="w") as config_file:
        json.dump(json.loads(config), config_file)
        config_file.flush()

        provider = custom_types.DataProviderParam()(config_file.name)
        assert isinstance(provider, getattr(providers, expected_provider_type))


def _endpoint_metadata(name: str, healthy: bool) -> EndpointMetadata:
    """
    Helper to build a basic EndpointMetadata with only name and healthy fields set
    """
    return EndpointMetadata(
        data={
            "endpoint-metadata": {"metadata": {"name": name}},
            "healthy": healthy,
            "endpoint": "/gordo/v0/test-project",
        }
    )


@pytest.mark.parametrize("tags", [["C", "A", "B", "D"], tu.SENSORS_STR_LIST])
def test_ml_server_dataframe_to_dict_and_back(tags: typing.List[str]):
    """
    Tests the flow of the server creating a dataframe from the model's data, putting into
    a dict of string to df. lists of values, and the client being able to reconstruct it back
    to the original dataframe (less the second level names)
    """
    # Some synthetic data
    original_input = np.random.random((10, len(tags)))
    model_output = np.random.random((10, len(tags)))

    # Convert this data into a dataframe with multi index columns
    df = model_utils.make_base_dataframe(tags, original_input, model_output)

    # Server then converts this into a dict which maps top level names to lists
    serialized = server_utils.dataframe_to_dict(df)

    # Client reproduces this dataframe
    df_clone = server_utils.dataframe_from_dict(serialized)

    # each subset of column under the top level names should be equal
    top_lvl_names = df.columns.get_level_values(0)
    for top_lvl_name in filter(lambda n: n not in ("start", "end"), top_lvl_names):
        assert np.allclose(df[top_lvl_name].values, df_clone[top_lvl_name].values)


@pytest.mark.parametrize(
    "endpoints,target,ignore_unhealthy,expected",
    [
        # One unhealthy + one healthy + no target + ignoring unhealthy = OK
        (
            [_endpoint_metadata("t1", False), _endpoint_metadata("t2", True)],
            None,
            True,
            [_endpoint_metadata("t2", True)],
        ),
        # One unhealthy + one healthy + no target + NOT ignoring unhealthy = ValueError
        (
            [_endpoint_metadata("t1", False), _endpoint_metadata("t2", True)],
            None,
            False,
            ValueError,
        ),
        # One unhealthy + one healthy + target (healthy) + do not ignore unhealthy = OK
        # because the client doesn't care about the unhealthy endpoint with a set target
        (
            [_endpoint_metadata("t1", False), _endpoint_metadata("t2", True)],
            "t2",
            False,
            [_endpoint_metadata("t2", True)],
        ),
        # All unhealthy = ValueError
        # ...even if we're suppose to ignore unhealthy endpoints, because Noen are healthy
        (
            [_endpoint_metadata("t1", False), _endpoint_metadata("t2", False)],
            None,
            True,
            ValueError,
        ),
        # All unhealthy + ignore unhealthy = ValueError
        # ...because we end up with no endpoints
        (
            [_endpoint_metadata("t1", False), _endpoint_metadata("t2", False)],
            None,
            True,
            ValueError,
        ),
        # All healthy = OK
        (
            [_endpoint_metadata("t1", True), _endpoint_metadata("t2", True)],
            None,
            True,
            [_endpoint_metadata("t1", True), _endpoint_metadata("t2", True)],
        ),
        # All healthy + target = OK (should get back only that target)
        (
            [_endpoint_metadata("t1", True), _endpoint_metadata("t2", True)],
            "t2",
            True,
            [_endpoint_metadata("t2", True)],
        ),
        # Want to filter down to one target, but that target is not healthy, ValueError
        (
            [_endpoint_metadata("t1", True), _endpoint_metadata("t2", False)],
            "t2",
            True,
            ValueError,
        ),
    ],
)
def test_client_endpoint_filtering(
    endpoints: typing.List[EndpointMetadata],
    target: typing.Optional[str],
    ignore_unhealthy: typing.Optional[bool],
    expected: typing.List[EndpointMetadata],
):
    if not isinstance(expected, list):
        with pytest.raises(ValueError):
            Client._filter_endpoints(endpoints, target, ignore_unhealthy)
    else:
        filtered_endpoints = Client._filter_endpoints(
            endpoints, target, ignore_unhealthy
        )
        assert (
            expected == filtered_endpoints
        ), f"Not equal: {expected} \n----\n {filtered_endpoints}"


def test_exponential_sleep_time(caplog, watchman_service):
    endpoint = _endpoint_metadata("t1", False)

    start, end = (
        isoparse("2016-01-01T00:00:00+00:00"),
        isoparse("2016-01-01T12:00:00+00:00"),
    )

    with caplog.at_level(logging.CRITICAL):
        with patch(
            "gordo_components.client.client.sleep", return_value=None
        ) as time_sleep:
            client = Client(project=tu.GORDO_PROJECT)

            client._send_prediction_request(
                X=pd.DataFrame([123]),
                y=None,
                chunk=slice(0, 1),
                endpoint=endpoint,
                start=start,
                end=end,
            )

            expected_calls = [call(8), call(16), call(32), call(64), call(128)]
            time_sleep.assert_has_calls(expected_calls)


def test__handle_response_errors():
    """
    Test expected error raising from gordo_components.client.io._handle_response
    """
    resp = requests.Response()
    resp.status_code = 422
    with pytest.raises(HttpUnprocessableEntity):
        _handle_response(resp)

    resp = requests.Response()
    resp.status_code = 403
    with pytest.raises(BadRequest):
        _handle_response(resp)

    resp = requests.Response()
    resp.status_code = 502
    with pytest.raises(IOError):
        _handle_response(resp)
