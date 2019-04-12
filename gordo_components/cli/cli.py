# -*- coding: utf-8 -*-

"""
CLI interfaces
"""

import logging
import ast
import os
from ast import literal_eval

import yaml
import click

from gordo_components import __version__
from gordo_components.builder.build_model import provide_saved_model
from gordo_components.data_provider.providers import (
    DataLakeProvider,
    InfluxDataProvider,
)
from gordo_components.server import server
from gordo_components import watchman
from gordo_components.cli.client import client as gordo_client
from gordo_components.dataset.sensor_tag import normalize_sensor_tags
from gordo_components.workflow.workflow_generator.workflow_generator import (
    machine_config_unique_tags_cli,
    workflow_generator_cli,
)

import dateutil.parser

# Set log level, defaulting to DEBUG
log_level = os.getenv("LOG_LEVEL", "DEBUG").upper()
azure_log_level = os.getenv("AZURE_DATALAKE_LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, log_level),
    format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
)
logger = logging.getLogger(__name__)
logging.getLogger("azure.datalake").setLevel(azure_log_level)


@click.group("gordo-components")
@click.version_option(version=__version__, message=__version__)
def gordo():
    """
    The main entry point for the CLI interface
    """
    pass


DEFAULT_MODEL_CONFIG = (
    "{'gordo_components.model.models.KerasAutoEncoder': {'kind': "
    "'feedforward_hourglass'}} "
)


@click.command()
@click.argument("output-dir", default="/data", envvar="OUTPUT_DIR")
@click.argument(
    "model-config", envvar="MODEL_CONFIG", default=DEFAULT_MODEL_CONFIG, type=yaml.load
)
@click.argument(
    "data-config",
    envvar="DATA_CONFIG",
    default='{"type": "TimeSeriesDataset"}',
    type=literal_eval,
)
@click.option("--metadata", envvar="METADATA", default="{}", type=literal_eval)
@click.option(
    "--model-register-dir",
    default=None,
    envvar="MODEL_REGISTER_DIR",
    type=click.Path(
        exists=False, file_okay=False, dir_okay=True, writable=True, readable=True
    ),
)
def build(output_dir, model_config, data_config, metadata, model_register_dir):
    """
    Build a model and deposit it into 'output_dir' given the appropriate config
    settings.

    \b
    Parameters
    ----------
    output_dir: str
        Directory to save model & metadata to.
    model_config: dict
        kwargs to be used in initializing the model. Should also
        contain kwarg 'type' which references the model to use. ie. KerasAutoEncoder
    data_config: dict
        kwargs to be used in intializing the dataset. Should also
        contain kwarg 'type' which references the dataset to use. ie. InfluxBackedDataset
    metadata: dict
        Any additional metadata to save under the key 'user-defined'
    model_register_dir: path
        Path to a directory which will index existing models and their locations, used
        for re-using old models instead of rebuilding them. If omitted then always
        rebuild
    """

    # TODO: Move all data related input from environment variable to data_config,
    # TODO: thereby removing all these data_config['variable'] lines

    data_config["tag_list"] = data_config.pop("tags")

    # TODO: Move parsing from here, into the InfluxDataSet class
    data_config["from_ts"] = dateutil.parser.isoparse(
        data_config.pop("train_start_date")
    )

    # TODO: Move parsing from here, into the InfluxDataSet class
    data_config["to_ts"] = dateutil.parser.isoparse(data_config.pop("train_end_date"))

    # Set default data provider for data config
    data_config["data_provider"] = DataLakeProvider()
    asset = data_config.get("asset", None)
    tag_list = normalize_sensor_tags(data_config["tag_list"], asset)

    data_config["tag_list"] = tag_list

    logger.info(f"Building, output will be at: {output_dir}")
    logger.info(f"Model config: {model_config}")
    logger.info(f"Data config: {data_config}")
    logger.info(f"Register dir: {model_register_dir}")

    model_location = provide_saved_model(
        model_config, data_config, metadata, output_dir, model_register_dir
    )
    with open("/tmp/model-location.txt", "w") as f:
        f.write(model_location)
    return 0


@click.command("run-server")
@click.option("--host", type=str, help="The host to run the server on.")
@click.option("--port", type=int, help="The port to run the server on.")
@click.option("--src-influx-host", type=str, envvar="SRC_INFLUXDB_HOST")
@click.option("--src-influx-port", type=int, envvar="SRC_INFLUXDB_PORT")
@click.option("--src-influx-username", type=str, envvar="SRC_INFLUXDB_USERNAME")
@click.option("--src-influx-password", type=str, envvar="SRC_INFLUXDB_PASSWORD")
@click.option("--src-influx-database", type=str, envvar="SRC_INFLUXDB_DATABASE")
@click.option("--src-influx-path", type=str, envvar="SRC_INFLUXDB_PATH")
@click.option("--src-influx-measurement", type=str, envvar="SRC_INFLUXDB_MEASUREMENT")
@click.option(
    "--src-influx-value", type=str, envvar="SRC_INFLUXDB_VALUE_NAME", default="value"
)
@click.option("--src-influx-api-key", type=str, envvar="SRC_INFLUXDB_API_KEY")
@click.option(
    "--src-influx-api-key-header", type=str, envvar="SRC_INFLUXDB_API_KEY_HEADER"
)
def run_server_cli(
    host: str,
    port: int,
    src_influx_host: str,
    src_influx_port: int,
    src_influx_username: str,
    src_influx_password: str,
    src_influx_database: str,
    src_influx_path: str,
    src_influx_measurement: str,
    src_influx_value: str,
    src_influx_api_key: str,
    src_influx_api_key_header: str,
):

    # We have have a hostname, then we make a data provider
    if src_influx_host:
        influx_config = {
            "host": src_influx_host,
            "port": src_influx_port,
            "username": src_influx_username,
            "password": src_influx_password,
            "database": src_influx_database,
            "proxies": {"http": "", "https": ""},
            "ssl": True,
            "path": src_influx_path,
            "timeout": 20,
            "retries": 10,
        }

        provider = InfluxDataProvider(
            measurement=src_influx_measurement,
            value_name=src_influx_value,
            api_key=src_influx_api_key,
            api_key_header=src_influx_api_key_header,
            **influx_config,
        )
    else:
        provider = None  # type: ignore

    server.run_server(host, port, data_provider=provider)


@click.command("run-watchman")
@click.argument("project-name", envvar="PROJECT_NAME", type=str)
@click.argument("project-version", envvar="PROJECT_VERSION", type=str)
@click.argument("target-names", envvar="TARGET_NAMES", type=ast.literal_eval)
@click.option(
    "--host", type=str, help="The host to run the server on.", default="0.0.0.0"
)
@click.option("--port", type=int, help="The port to run the server on.", default=5555)
@click.option("--debug", type=bool, help="Run in debug mode", default=False)
@click.option(
    "--namespace",
    type=str,
    help="Namespace watchman should make requests in for ML servers",
    default="kubeflow",
    envvar="NAMESPACE",
)
@click.option(
    "--ambassador-namespace",
    type=str,
    help="Namespace watchman expects Ambassador to be in",
    default="ambassador",
    envvar="AMBASSADOR_NAMESPACE",
)
def run_watchman_cli(
    project_name,
    project_version,
    target_names,
    host,
    port,
    debug,
    namespace,
    ambassador_namespace,
):
    """
    Start the Gordo Watchman server for this project. Which is responsible
    for dynamically comparing expected URLs derived from a project config fle
    against those actually deployed to determine and report their health.

    \b
    Must have the following environment variables set:
        PROJECT_NAME: project_name for the config file
        TARGET_NAMES: A list of non-sanitized machine / target names
    """
    watchman.server.run_server(
        host,
        port,
        debug,
        project_name,
        project_version,
        target_names,
        namespace=namespace,
        ambassador_namespace=ambassador_namespace,
    )


gordo.add_command(build)
gordo.add_command(run_server_cli)
gordo.add_command(run_watchman_cli)
gordo.add_command(gordo_client)
gordo.add_command(workflow_generator_cli)
gordo.add_command(machine_config_unique_tags_cli)


if __name__ == "__main__":
    gordo()
