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
from gordo_components.machine import Machine
from gordo_components.client.io import (
    _handle_response,
    HttpUnprocessableEntity,
    BadRequest,
)
from gordo_components.client.forwarders import ForwardPredictionsIntoInflux
from gordo_components.client.utils import PredictionResult
from gordo_components.machine.dataset.data_provider import providers
from gordo_components.server import utils as server_utils
from gordo_components.machine.model import utils as model_utils
from gordo_components import cli, serializer
from gordo_components.cli import custom_types

from tests import utils as tu


def test_client_get_metadata(ml_server):
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


def test_client_predict_specific_targets(ml_server):
    """
    Client.predict should filter any endpoints given to it.
    """
    client = Client(project=tu.GORDO_PROJECT)
    with mock.patch.object(
        Client,
        "predict_single_machine",
        return_value=PredictionResult("test-name", [], []),
    ) as patched:

        start = (isoparse("2016-01-01T00:00:00+00:00"),)
        end = isoparse("2016-01-01T12:00:00+00:00")

        # Should not actually call any predictions because this machine name doesn't exist
        client.predict(start=start, end=end, machine_names=["non-existant-machine"])
        patched.assert_not_called()

        # Should be called once, for this machine.
        client.predict(start=start, end=end, machine_names=[tu.GORDO_SINGLE_TARGET])
        patched.assert_called_once()


def test_client_download_model(ml_server):
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
    influxdb, ml_server, batch_size: int, use_parquet: bool
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

    # Should have discovered machine-1 and machine-2
    # defined in the example data of controller's response for /models/<project-name>
    assert len(prediction_client.machines) == 1

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


def test_client_cli_metadata(ml_server, tmp_dir):
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


def test_client_cli_download_model(ml_server, tmp_dir):
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
    ml_server,
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
    with patch(
        "gordo_components.machine.dataset.sensor_tag._asset_from_tag_name",
        side_effect=lambda *args, **kwargs: "default",
    ):
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
    should_fail, start_date, end_date, caplog, influxdb, ml_server
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
        with patch(
            "gordo_components.machine.dataset.sensor_tag._asset_from_tag_name",
            side_effect=lambda *args, **kwargs: "default",
        ):
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


def _machine(name: str) -> Machine:
    """
    Helper to build a basic Machine, only defining its name
    """
    from gordo_components.machine.dataset.sensor_tag import SensorTag

    return Machine.from_config(
        config={
            "name": name,
            "dataset": {
                "tag_list": [SensorTag("tag-1", "foo"), SensorTag("tag-2", "foo")],
                "train_start_date": "2016-01-01T00:00:00Z",
                "train_end_date": "2016-01-05T00:00:00Z",
            },
            "model": "sklearn.linear_model.LinearRegression",
        },
        project_name="test-project",
    )


@pytest.mark.parametrize(
    "machines,target,expected",
    [
        # Two machines, no target, should give two machines
        ([_machine("t1"), _machine("t2")], None, [_machine("t1"), _machine("t2")]),
        # One machine target should filter down to that machine
        ([_machine("t1"), _machine("t2")], "t2", [_machine("t2")]),
        # Target which doesn't match any machines raises error
        ([_machine("t1"), _machine("t2")], "t3", ValueError),
    ],
)
def test_client_machine_filtering(
    machines: typing.List[Machine],
    target: typing.Optional[str],
    expected: typing.List[Machine],
):
    if not isinstance(expected, list):
        with pytest.raises(ValueError):
            Client._filter_machines(machines, target)
    else:
        filtered_machines = Client._filter_machines(machines, target)
        assert (
            expected == filtered_machines
        ), f"Not equal: {expected} \n----\n {filtered_machines}"


def test_exponential_sleep_time(caplog, ml_server):

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
                machine=_machine("t1"),
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


@pytest.mark.parametrize("revision", (None, tu.GORDO_REVISION, "does-not-exist"))
def test_client_set_revision(ml_server, gordo_project, revision):
    """
    Client will auto set and verify revision
    """

    # Default behavior is to lookup the latest revision.
    if revision is None:
        client = Client(project=gordo_project, revision=revision)
        assert client.revision == tu.GORDO_REVISION
        assert client.session.headers["revision"] == tu.GORDO_REVISION

    # If we ask for a specific one, it should result in tha one.
    elif revision == tu.GORDO_REVISION:
        client = Client(project=gordo_project, revision=revision)
        assert client.revision == tu.GORDO_REVISION
        assert client.session.headers["revision"] == tu.GORDO_REVISION

    # Asking for that which doesn't exist will raise an error.
    else:
        with pytest.raises(LookupError):
            Client(project=gordo_project, revision=revision)


def test_client_auto_update_revision(ml_server, gordo_project, gordo_revision):
    """
    Given a client starts with a revision which is outdated, it will automatically update
    itself to match the latest being served.
    """
    client = Client(project=gordo_project)
    assert client.revision == gordo_revision  # by default it figures out the latest.

    # Abuse the private variable to change it to something else.
    client.session.headers["revision"] = "bad-revision"
    client._revision = "bad-revision"
    assert client.revision == "bad-revision"

    # Contacting the server with that revision will make the client update its revision
    client.get_machines()
    assert client.revision == gordo_revision
    assert client.session.headers["revision"] == gordo_revision
