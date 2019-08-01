# -*- coding: utf-8 -*-

import os
import tempfile
import json
import logging
import typing
from dateutil.parser import isoparse  # type: ignore

import aiohttp
import pytest
import pandas as pd
import numpy as np
from click.testing import CliRunner
from sklearn.base import BaseEstimator

from gordo_components.client import Client, utils as client_utils
from gordo_components.client.utils import EndpointMetadata
from gordo_components.client import io as client_io
from gordo_components.client.client import dataframe_from_dict_with_list_values
from gordo_components.client.forwarders import ForwardPredictionsIntoInflux
from gordo_components.data_provider import providers
from gordo_components.server.views.base import BaseModelView
from gordo_components import cli, serializer
from gordo_components.cli import custom_types

from tests import utils as tu


@pytest.mark.asyncio
@pytest.mark.parametrize("session", (True, False))
@pytest.mark.parametrize("timeout", (5, None))
@pytest.mark.parametrize("params", ({"key": "value"}, {}))
async def test_client_fetch_json(httpbin, session, timeout, params):
    """
    Test fetch_json accepts specific kwargs
    """
    session = aiohttp.ClientSession() if session else None
    resp = await client_io.fetch_json(
        f"http://{httpbin}/get", session=session, timeout=timeout, params=params
    )
    assert params == resp["args"]
    if session:
        await session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("session", (True, False))
@pytest.mark.parametrize("timeout", (5, None))
@pytest.mark.parametrize("json", ({"key": "value"}, {}))
async def test_client_post_json(httpbin, session, timeout, json):
    """
    Test post_json accepts specific kwargs
    """
    session = aiohttp.ClientSession() if session else None
    resp = await client_io.post_json(
        f"http://{httpbin}/post", session=session, timeout=timeout, json=json
    )
    assert json == resp["json"]
    if session:
        await session.close()


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
@pytest.mark.parametrize("use_data_provider", (False, True))
def test_client_predictions_diff_batch_sizes_and_toggle_data_provider(
    influxdb, watchman_service, use_data_provider: bool, batch_size: int
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

    data_provider = (
        providers.InfluxDataProvider(
            measurement=tu.INFLUXDB_MEASUREMENT,
            value_name="Value",
            client=client_utils.influx_client_from_uri(
                uri=tu.INFLUXDB_URI, dataframe_client=True
            ),
        )
        if use_data_provider
        else None
    )

    prediction_client = Client(
        project=tu.GORDO_PROJECT,
        data_provider=data_provider,
        prediction_forwarder=ForwardPredictionsIntoInflux(
            destination_influx_uri=tu.INFLUXDB_URI
        ),
        batch_size=batch_size,
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


def test_client_cli_metadata(watchman_service):
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
    with tempfile.TemporaryDirectory() as tmp_dir:
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


def test_client_cli_download_model(watchman_service):
    """
    Test proper execution of client predict sub-command
    """
    runner = CliRunner()

    with tempfile.TemporaryDirectory() as output_dir:

        # Empty output directory before downloading
        assert len(os.listdir(output_dir)) == 0

        out = runner.invoke(
            cli.gordo,
            args=[
                "client",
                "--project",
                tu.GORDO_PROJECT,
                "--target",
                tu.GORDO_SINGLE_TARGET,
                "download-model",
                output_dir,
            ],
        )
        assert (
            out.exit_code == 0
        ), f"Expected output code 0 got '{out.exit_code}', {out.output}"

        # Output directory should not be empty any longer
        assert len(os.listdir(output_dir)) > 0

        model_output_dir = os.path.join(output_dir, tu.GORDO_SINGLE_TARGET)
        assert os.path.isdir(model_output_dir)

        model = serializer.load(model_output_dir)
        assert isinstance(model, BaseEstimator)


@pytest.mark.parametrize(
    "forwarder_args",
    [["--influx-uri", tu.INFLUXDB_URI, "--forward-resampled-sensors"], None],
)
@pytest.mark.parametrize("output_dir", [tempfile.TemporaryDirectory(), None])
@pytest.mark.parametrize("data_provider", [providers.RandomDataProvider(), None])
def test_client_cli_predict(
    influxdb, watchman_service, forwarder_args, output_dir, data_provider
):
    """
    Test ability for client to get predictions via CLI
    """
    runner = CliRunner()

    args = [
        "client",
        "--metadata",
        "key,value",
        "--project",
        tu.GORDO_PROJECT,
        "predict",
        "2016-01-01T00:00:00Z",
        "2016-01-01T01:00:00Z",
    ]

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
    if output_dir is not None:
        args.extend(["--output-dir", output_dir.name])

    # Do we have a data provider, POST else GET requests
    if data_provider is not None:
        args.extend(["--data-provider", json.dumps(data_provider.to_dict())])

    # Run without any error
    out = runner.invoke(cli.gordo, args=args)
    assert out.exit_code == 0, f"{out.output}"

    # If we activated forwarder and we had any actual data then there should
    # be resampled values in the influx
    if forwarder_args and data_provider:
        vals = influx_client.query(query)
        assert len(vals) == 1
        assert len(vals["resampled"]) == 28
        influx_client.drop_measurement("resampled")

    # Did it save dataframes to output dir if specified?
    if output_dir is not None:
        assert os.path.exists(
            os.path.join(output_dir.name, f"{tu.GORDO_SINGLE_TARGET}.csv.gz")
        )


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
        assert out.exit_code != 0, f"{out.output}"
    else:
        assert out.exit_code == 0, f"{out.output}"


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
        target_name=name, healthy=healthy, endpoint=None, tag_list=None, resolution=None
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
    transformed_model_input = np.random.random((10, len(tags)))
    inverse_transformed_model_output = np.random.random((10, len(tags)))

    # Convert this data into a dataframe with multi index columns
    df = BaseModelView.make_base_dataframe(
        tags,
        original_input,
        model_output,
        transformed_model_input,
        inverse_transformed_model_output,
    )

    # Server then converts this into a dict which maps top level names to lists
    as_dict_to_list_vals = BaseModelView.multi_lvl_column_dataframe_to_dict(df)

    # Client reproduces this dataframe
    df_clone = dataframe_from_dict_with_list_values(as_dict_to_list_vals)

    # each subset of column under the top level names should be equal
    for top_lvl_name in df.columns.get_level_values(0):
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
