# -*- coding: utf-8 -*-

import os
import typing
import sys
import json
from datetime import datetime
from pprint import pprint

import click
import pandas as pd  # noqa

from gordo_components.client import Client
from gordo_components import serializer
from gordo_components.data_provider import providers
from gordo_components.cli.custom_types import (
    IsoFormatDateTime,
    DataProviderParam,
    key_value_par,
)
from gordo_components.client.forwarders import ForwardPredictionsIntoInflux


@click.group("client")
@click.option("--project", help="The project to target")
@click.option("--target", help="Single target, instead of all project targets")
@click.option("--host", help="The host the server is running on", default="localhost")
@click.option("--port", help="Port the server is running on", default=443)
@click.option("--scheme", help="tcp/http/https", default="https")
@click.option("--gordo-version", help="Version of gordo", default="v0")
@click.option("--batch-size", help="How many samples to send", default=100000)
@click.option("--parallelism", help="Maximum asynchronous jobs to run", default=10)
@click.option(
    "--metadata",
    type=key_value_par,
    multiple=True,
    default=(),
    help="Key-Value pair to be entered as metadata labels, may use this option multiple times. "
    "to be separated by a comma. ie: --metadata key,val --metadata some key,some value",
)
@click.pass_context
def client(ctx: click.Context, *args, **kwargs):
    """
    Entry sub-command for client related activities
    """
    ctx.obj = {"args": args, "kwargs": kwargs}


@click.command("predict")
@click.argument("start", type=IsoFormatDateTime())
@click.argument("end", type=IsoFormatDateTime())
@click.option(
    "--data-provider",
    type=DataProviderParam(),
    envvar="DATA_PROVIDER",
    help="DataProvider dict encoded as json. Must contain a 'type' key with the name of"
    " a DataProvider as value.",
)
@click.option(
    "--output-dir",
    type=click.Path(exists=True),
    help="Save output prediction dataframes in a directory",
)
@click.option(
    "--influx-uri",
    help="Format: <username>:<password>@<host>:<port>/<optional-path>/<db_name>",
)
@click.option("--influx-api-key", help="Key to provide to the destination influx")
@click.option(
    "--influx-recreate-db",
    help="Recreate the desintation DB before writing",
    is_flag=True,
    default=False,
)
@click.option(
    "--forward-resampled-sensors",
    help="forward the resampled sensor values",
    is_flag=True,
    default=False,
)
@click.option(
    "--ignore-unhealthy-targets",
    help="Ignore any unhealthy targets. By default the client will raise an "
    "error if any unhealthy endpoints are encountered.",
    is_flag=True,
    default=False,
)
@click.option(
    "--n-retries",
    help="Time client should retry failed predictions",
    type=int,
    default=5,
)
@click.option(
    "--parquet/--no-parquet",
    help="Use parquet serialization when sending and receiving data from server",
    default=True,
)
@click.pass_context
def predict(
    ctx: click.Context,
    start: datetime,
    end: datetime,
    data_provider: providers.GordoBaseDataProvider,
    output_dir: str,
    influx_uri: str,
    influx_api_key: str,
    influx_recreate_db: bool,
    forward_resampled_sensors: bool,
    ignore_unhealthy_targets: bool,
    n_retries: int,
    parquet: bool,
):
    """
    Run some predictions against the target
    """
    ctx.obj["kwargs"].update(
        {
            "data_provider": data_provider,
            "forward_resampled_sensors": forward_resampled_sensors,
            "ignore_unhealthy_targets": ignore_unhealthy_targets,
            "n_retries": n_retries,
            "use_parquet": parquet,
        }
    )

    client = Client(*ctx.obj["args"], **ctx.obj["kwargs"])

    if influx_uri is not None:
        client.prediction_forwarder = ForwardPredictionsIntoInflux(
            destination_influx_uri=influx_uri,
            destination_influx_api_key=influx_api_key,
            destination_influx_recreate=influx_recreate_db,
            n_retries=n_retries,
        )

    # Fire off getting predictions
    predictions = client.predict(
        start, end
    )  # type: typing.Iterable[typing.Tuple[str, pd.DataFrame, typing.List[str]]]

    # Loop over all error messages for each result and log them
    click.secho(f"\n{'-' * 20} Summary of failed predictions (if any) {'-' * 20}")
    exit_code = 0
    for (_name, _df, error_messages) in predictions:
        for err_msg in error_messages:
            # Any error message indicates we encountered at least one error
            exit_code = 1
            click.secho(err_msg, fg="red")

    # Shall we write the predictions out?
    if output_dir is not None:
        for (name, prediction_df, _err_msgs) in predictions:
            prediction_df.to_csv(
                os.path.join(output_dir, f"{name}.csv.gz"), compression="gzip"
            )
    sys.exit(exit_code)


@click.command("metadata")
@click.option(
    "--output-file",
    type=click.File(mode="w"),
    help="Optional output file to save metadata",
)
@click.pass_context
def metadata(ctx: click.Context, output_file: typing.Optional[typing.IO[str]]):
    """
    Get metadata from a given endpoint
    """
    client = Client(*ctx.obj["args"], **ctx.obj["kwargs"])
    metadata = client.get_metadata()
    if output_file:
        json.dump(metadata, output_file)
        click.secho(f"Saved metadata json to file: '{output_file}'")
    else:
        pprint(metadata)
    return metadata


@click.command("download-model")
@click.argument("output-dir", type=click.Path(exists=True))
@click.pass_context
def download_model(ctx: click.Context, output_dir: str):
    """
    Download the actual model from the target and write to an output directory
    """
    client = Client(*ctx.obj["args"], **ctx.obj["kwargs"])
    models = client.download_model()

    # Iterate over mapping of models and save into their own sub dirs of the output_dir
    for target, model in models.items():
        model_out_dir = os.path.join(output_dir, target)
        os.mkdir(model_out_dir)
        click.secho(
            f"Writing model '{target}' to directory: '{model_out_dir}'...", nl=False
        )
        serializer.dump(model, model_out_dir)
        click.secho(f"done")

    click.secho(f"Wrote all models to directory: {output_dir}", fg="green")


client.add_command(predict)
client.add_command(metadata)
client.add_command(download_model)
