# -*- coding: utf-8 -*-

import os
import tempfile
import json
from contextlib import ExitStack
from dateutil.parser import isoparse  # type: ignore

import aiohttp
import pytest
import pandas as pd
from click.testing import CliRunner
from sklearn.base import BaseEstimator

from gordo_components.client import Client, utils as client_utils
from gordo_components.client import io as client_io
from gordo_components.client.forwarders import ForwardPredictionsIntoInflux
from gordo_components.data_provider import providers
from gordo_components import cli, serializer
from gordo_components.cli import custom_types

from tests.gordo_components.server.test_gordo_server import influxdatabase, SENSORS
from tests.utils import watchman


@pytest.mark.asyncio
@pytest.mark.parametrize("session", (True, False))
@pytest.mark.parametrize("timeout", (5, None))
@pytest.mark.parametrize("params", ({"key": "value"}, {}))
async def test_client_fetch_json(session, timeout, params):
    """
    Test fetch_json accepts specific kwargs
    """
    session = aiohttp.ClientSession() if session else None
    resp = await client_io.fetch_json(
        "http://httpbin.org/get", session=session, timeout=timeout, params=params
    )
    assert params == resp["args"]
    if session:
        await session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("session", (True, False))
@pytest.mark.parametrize("timeout", (5, None))
@pytest.mark.parametrize("json", ({"key": "value"}, {}))
async def test_client_post_json(session, timeout, json):
    """
    Test post_json accepts specific kwargs
    """
    session = aiohttp.ClientSession() if session else None
    resp = await client_io.post_json(
        "http://httpbin.org/post", session=session, timeout=timeout, json=json
    )
    assert json == resp["json"]
    if session:
        await session.close()


def test_client_get_metadata(trained_model_directory: pytest.fixture):
    """
    Test client's ability to get metadata from some target
    """

    with watchman(
        host="localhost",
        project="gordo-test",
        targets=["machine-1"],
        model_location=trained_model_directory,
    ):
        client = Client(project="gordo-test")

        metadata = client.get_metadata()
        assert isinstance(metadata, dict)

        # Can't get metadata for non-existent target
        with pytest.raises(ValueError):
            client = Client(project="gordo-test", target="no-such-target")
            client.get_metadata()


def test_client_download_model(trained_model_directory: pytest.fixture):
    """
    Test client's ability to download the model
    """

    with watchman(
        host="localhost",
        project="gordo-test",
        targets=["machine-1"],
        model_location=trained_model_directory,
    ):
        client = Client(project="gordo-test", target="machine-1")

        models = client.download_model()
        assert isinstance(models, dict)
        assert isinstance(models["machine-1"], BaseEstimator)

        # Can't download model for non-existent target
        with pytest.raises(ValueError):
            client = Client(project="gordo-test", target="machine-2")
            client.download_model()


@pytest.mark.dockertest
@pytest.mark.parametrize("use_data_provider", (True, False))
@pytest.mark.parametrize("trained_model_directory", (SENSORS,), indirect=True)
def test_client_predictions_with_or_without_data_provider(
    trained_model_directory: pytest.fixture, use_data_provider: bool
):
    """
    Run the prediction client with or without a data provider
    """

    with watchman(
        host="localhost",
        project="gordo-test",
        targets=["machine-1"],
        model_location=trained_model_directory,
    ), influxdatabase(
        sensors=SENSORS,
        db_name="testdb",
        user="root",
        password="root",
        measurement="sensors",
    ):

        # The uri for the local influx which serves as the source and destination
        uri = f"root:root@localhost:8086/testdb"

        # Time range used in this test
        start, end = isoparse("2016-01-01T00:00:00Z"), isoparse("2016-01-01T12:00:00Z")

        # Client only used within the this test
        test_client = client_utils.influx_client_from_uri(uri)

        # Created measurements by prediction client with dest influx
        output_measurements = ("predictions", "anomaly")
        query_tmpl = """
        SELECT *
        FROM "{measurement}"
        WHERE("machine" =~ /^machine-1$/)
        """

        # Before predicting, influx destination db should be empty for 'predictions' measurement
        for measurement in output_measurements:
            vals = test_client.query(query_tmpl.format(measurement=measurement))
            assert len(vals) == 0

        data_provider = (
            providers.InfluxDataProvider(
                measurement="sensors",
                value_name="Value",
                client=client_utils.influx_client_from_uri(
                    uri=uri, dataframe_client=True
                ),
            )
            if use_data_provider
            else None
        )

        prediction_client = Client(
            project="gordo-test",
            data_provider=data_provider,
            prediction_forwarder=ForwardPredictionsIntoInflux(
                destination_influx_uri=uri
            ),
        )

        # Should have discovered machine-1 & machine-2
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
        for measurement in output_measurements:
            vals = test_client.query(query_tmpl.format(measurement=measurement))
            assert (
                len(vals) > 0
            ), f"Expected new values in '{measurement}' measurement, but found {vals}"


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


def test_client_cli_metadata(trained_model_directory: pytest.fixture):
    """
    Test proper execution of client predict sub-command
    """
    runner = CliRunner()

    with watchman(
        host="localhost",
        project="gordo-test",
        targets=["machine-1"],
        model_location=trained_model_directory,
    ):

        # Simple metadata fetching
        out = runner.invoke(
            cli.gordo,
            args=[
                "client",
                "--project",
                "gordo-test",
                "--target",
                "machine-1",
                "metadata",
            ],
        )
        assert out.exit_code == 0
        assert "machine-1" in out.output

        # Save metadata to file
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_file = os.path.join(tmp_dir, "metadata.json")
            out = runner.invoke(
                cli.gordo,
                args=[
                    "client",
                    "--project",
                    "gordo-test",
                    "--target",
                    "machine-1",
                    "metadata",
                    "--output-file",
                    output_file,
                ],
            )
            assert out.exit_code == 0, f"{out.exc_info}"
            assert os.path.exists(output_file)
            with open(output_file) as f:
                metadata = json.load(f)
                assert "machine-1" in metadata


def test_client_cli_download_model(
    trained_model_directory: pytest.fixture, tmp_dir: pytest.fixture
):
    """
    Test proper execution of client predict sub-command
    """
    runner = CliRunner()

    with watchman(
        host="localhost",
        project="gordo-test",
        targets=["machine-1"],
        model_location=trained_model_directory,
    ), tempfile.TemporaryDirectory() as output_dir:

        # Empty output directory before downloading
        assert len(os.listdir(output_dir)) == 0

        out = runner.invoke(
            cli.gordo,
            args=[
                "client",
                "--project",
                "gordo-test",
                "--target",
                "machine-1",
                "download-model",
                output_dir,
            ],
        )
        assert (
            out.exit_code == 0
        ), f"Expected output code 0 got '{out.exit_code}', {out.output}"

        # Output directory should not be empty any longer
        assert len(os.listdir(output_dir)) > 0

        model_output_dir = os.path.join(output_dir, "machine-1")
        assert os.path.isdir(model_output_dir)

        model = serializer.load(model_output_dir)
        assert isinstance(model, BaseEstimator)


@pytest.mark.dockertest
@pytest.mark.parametrize(
    "forwarder_args", [["--influx-uri", "root:root@localhost:8086/sensors"], None]
)
@pytest.mark.parametrize("output_dir", [tempfile.TemporaryDirectory(), None])
@pytest.mark.parametrize("data_provider", [providers.RandomDataProvider(), None])
def test_client_cli_predict(
    trained_model_directory: pytest.fixture, forwarder_args, output_dir, data_provider
):
    """

    """
    runner = CliRunner()

    with ExitStack() as stack:

        # Always need watchman
        stack.enter_context(
            watchman(
                host="localhost",
                project="gordo-test",
                targets=["machine-1"],
                model_location=trained_model_directory,
            )
        )

        # Might need influx
        if forwarder_args:
            stack.enter_context(
                influxdatabase(
                    sensors=SENSORS, db_name="sensors", user="root", password="root"
                )
            )

        args = [
            "client",
            "--metadata",
            "key,value",
            "--project",
            "gordo-test",
            "predict",
            "2016-01-01T00:00:00Z",
            "2016-01-01T01:00:00Z",
        ]

        # Do we have forwarder args?
        if forwarder_args is not None:
            args.extend(forwarder_args)

        # Should it write out the predictions to dataframes in an output directory?
        if output_dir is not None:
            args.extend(["--output-dir", output_dir.name])

        # Do we have a data provider, POST else GET requests
        if data_provider is not None:
            args.extend(["--data-provider", json.dumps(data_provider.to_dict())])

        # Run without any error
        out = runner.invoke(cli.gordo, args=args)
        assert out.exit_code == 0, f"{out.output}"

        # Did it save dataframes to output dir if specified?
        if output_dir is not None:
            assert os.path.exists(os.path.join(output_dir.name, "machine-1.csv.gz"))


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
