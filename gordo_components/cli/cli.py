# -*- coding: utf-8 -*-

"""
CLI interfaces
"""

import logging
import os

import jinja2
import yaml
import click
from gordo_components.builder.build_model import provide_saved_model
from gordo_components.data_provider.providers import (
    DataLakeProvider,
    InfluxDataProvider,
)
from gordo_components.serializer import (
    load_metadata,
    pipeline_into_definition,
    pipeline_from_definition,
)
from gordo_components.server import server
from gordo_components import watchman
from gordo_components.cli.client import client as gordo_client
from gordo_components.cli.custom_types import key_value_par
from gordo_components.dataset.sensor_tag import normalize_sensor_tags

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
@click.argument("name", envvar="MODEL_NAME")
@click.argument("output-dir", default="/data", envvar="OUTPUT_DIR")
@click.argument(
    "model-config", envvar="MODEL_CONFIG", default=DEFAULT_MODEL_CONFIG, type=str
)
@click.argument(
    "data-config",
    envvar="DATA_CONFIG",
    default='{"type": "TimeSeriesDataset"}',
    type=yaml.safe_load,
)
@click.option("--metadata", envvar="METADATA", default="{}", type=yaml.safe_load)
@click.option(
    "--model-register-dir",
    default=None,
    envvar="MODEL_REGISTER_DIR",
    type=click.Path(
        exists=False, file_okay=False, dir_okay=True, writable=True, readable=True
    ),
)
@click.option(
    "--print-cv-scores", help="Prints CV scores to stdout", is_flag=True, default=False
)
@click.option(
    "--model-parameter",
    type=key_value_par,
    multiple=True,
    default=(),
    help="Key-Value pair for a model parameter and its value, may use this option "
    "multiple times. Separate key,valye by a comma. ie: --model-parameter key,val "
    "--model-parameter some_key,some_value",
)
@click.option(
    "--model-location-file",
    help="Full path to a file to create and write the location of where the serialized model is placed.",
    type=click.File(mode="w", lazy=False),
    default="/tmp/model-location.txt",
)
@click.option(
    "--data-provider-threads",
    help="Number of threads to use for the data provider when fetching data",
    envvar="DATA_PROVIDER_THREADS",
    type=int,
    default=1,
)
def build(
    name,
    output_dir,
    model_config,
    data_config,
    metadata,
    model_register_dir,
    print_cv_scores,
    model_parameter,
    model_location_file,
    data_provider_threads,
):
    """
    Build a model and deposit it into 'output_dir' given the appropriate config
    settings.

    \b
    Parameters
    ----------
    name: str
        Name given to the model to build
    output_dir: str
        Directory to save model & metadata to.
    model_config: str
        String containing a yaml which will be parsed to a dict which will be used in
        initializing the model. Should also contain key 'type' which references the
        model to use. ie. KerasAutoEncoder
    data_config: dict
        kwargs to be used in intializing the dataset. Should also
        contain kwarg 'type' which references the dataset to use. ie. InfluxBackedDataset
    metadata: dict
        Any additional metadata to save under the key 'user-defined'
    model_register_dir: path
        Path to a directory which will index existing models and their locations, used
        for re-using old models instead of rebuilding them. If omitted then always
        rebuild
    print_cv_scores: bool
        Print cross validation scores to stdout
    model_parameter: List[Tuple]
        List of model key-values, wheres the values will be injected into the model
        config wherever there is a jinja variable with the key.
    model_location_file: str/path
        Path to a file to open and write the location of the serialized model to.
    data_provider_threads: int
        Number of threads to use for the data provider when fetching data.
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
    data_config["data_provider"] = DataLakeProvider(threads=data_provider_threads)
    asset = data_config.get("asset", None)
    tag_list = normalize_sensor_tags(data_config["tag_list"], asset)

    data_config["tag_list"] = tag_list

    # Normalize target tag list if present
    if "target_tag_list" in data_config:
        target_tag_list = normalize_sensor_tags(data_config["target_tag_list"], asset)
        data_config["target_tag_list"] = target_tag_list

    logger.info(f"Building, output will be at: {output_dir}")
    logger.info(f"Raw model config: {model_config}")
    logger.info(f"Data config: {data_config}")
    logger.info(f"Register dir: {model_register_dir}")

    model_parameter = dict(model_parameter)
    model_config = expand_model(model_config, model_parameter)
    model_config = yaml.full_load(model_config)

    # Convert the config into a pipeline, and back into definition to ensure
    # all default parameters are part of the config.
    logger.debug(f"Ensuring the passed model config is fully expanded.")
    model_config = pipeline_into_definition(pipeline_from_definition(model_config))
    logger.debug(f"Fully expanded model config: {model_config}")

    model_location = provide_saved_model(
        name, model_config, data_config, metadata, output_dir, model_register_dir
    )
    # If the model is cached but without CV scores then we force a rebuild. We do this
    # by deleting the entry in the cache and then rerun `provide_saved_model`
    # (leaving the old model laying around)
    if print_cv_scores:
        saved_metadata = load_metadata(model_location)
        all_scores = get_all_score_strings(saved_metadata)
        if not all_scores:
            logger.warning(
                "Found that loaded model does not have cross validation values "
                "even though we were asked to print them, clearing cache and "
                "rebuilding model"
            )

            model_location = provide_saved_model(
                name,
                model_config,
                data_config,
                metadata,
                output_dir,
                model_register_dir,
                replace_cache=True,
            )
            saved_metadata = load_metadata(model_location)
            all_scores = get_all_score_strings(saved_metadata)

        for score in all_scores:
            print(score)

    # Write out the model location to this file.
    model_location_file.write(model_location)
    return 0


def expand_model(model_config: str, model_parameters: dict):
    """
    Expands the jinja template which is the model using the variables in
    `model_parameters`

    Parameters
    ----------
    model_config: str
        Jinja template which when expanded becomes a valid model config json.
    model_parameters:
        Parameters for the model config.

    Raises
    ------
    ValueError
        If an undefined variable is used in the model_config.

    Returns
    -------
    str
        The model config with variables expanded

    """
    try:
        model_template = jinja2.Environment(
            loader=jinja2.BaseLoader(), undefined=jinja2.StrictUndefined
        ).from_string(model_config)
        model_config = model_template.render(**model_parameters)
    except jinja2.exceptions.UndefinedError as e:
        raise ValueError("Model parameter missing value!") from e
    logger.info(f"Expanded model config: {model_config}")
    return model_config


def get_all_score_strings(metadata):
    """Given metadata from the model builder this function returns a list of
    strings of the format {metric_name}_{score_name}={score_val} for katib
    to pick up. Replaces all spaces with `-`.

    Parameters
    ----------
    metadata : dict
        Metadata dictionary. Must contain a dictionary in
        metadata.model.cross-validation.scores with at least one metric as key and
        value being another map with score key/values. See example

    Examples
    --------
    >>> score_strings = get_all_score_strings(
    ...  {
    ...     "model": {
    ...         "cross-validation": {
    ...             "scores": {"explained variance": {"min": 0, "max": 2}}
    ...         }
    ...     }
    ...   }
    ... )
    >>> len(score_strings)
    2
    >>> score_strings
    ['explained-variance_min=0', 'explained-variance_max=2']


    """
    all_scores = []
    for metric_name, scores in (
        metadata.get("model", dict())
        .get("cross-validation", dict())
        .get("scores", dict())
        .items()
    ):
        metric_name = metric_name.replace(" ", "-")
        for score_name, score_val in scores.items():
            score_name = score_name.replace(" ", "-")
            all_scores.append(f"{metric_name}_{score_name}={score_val}")
    return all_scores


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
@click.argument("target-names", envvar="TARGET_NAMES", type=yaml.safe_load)
@click.option(
    "--host", type=str, help="The host to run the server on.", default="0.0.0.0"
)
@click.option("--port", type=int, help="The port to run the server on.", default=5555)
@click.option("--debug", type=bool, help="Run in debug mode.", default=False)
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
    help="Namespace watchman expects Ambassador to be in.",
    default="ambassador",
    envvar="AMBASSADOR_NAMESPACE",
)
@click.option(
    "--ambassador-host",
    type=str,
    help="Full hostname of ambassador. If this is set then `--ambassador-namespace` is "
    "ignored even if set explicitly.",
    default=None,
    envvar="AMBASSADOR_HOST",
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
    ambassador_host,
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
        ambassador_host=ambassador_host
        if ambassador_host
        else f"ambassador.{ambassador_namespace}",
    )


gordo.add_command(build)
gordo.add_command(run_server_cli)
gordo.add_command(run_watchman_cli)
gordo.add_command(gordo_client)

if __name__ == "__main__":
    gordo()
