# -*- coding: utf-8 -*-

"""
CLI interfaces
"""

import logging
import sys
import traceback

from gordo_components.machine.dataset.data_provider.providers import (
    NoSuitableDataProviderError,
)
from gordo_components.machine.dataset.sensor_tag import SensorTagNormalizationError
from gordo_components.machine.dataset.datasets import InsufficientDataError
from gordo_components.builder.mlflow_utils import MlflowLoggingError
from gunicorn.glogging import Logger

import jinja2
import yaml
import click
from typing import Dict, Type, Tuple, List, Any

from gordo_components.builder.build_model import ModelBuilder
from gordo_components import serializer
from gordo_components.builder import mlflow_utils
from gordo_components.server import server
from gordo_components import __version__
from gordo_components.machine import Machine
from gordo_components.cli.workflow_generator import workflow_cli
from gordo_components.cli.client import client as gordo_client
from gordo_components.cli.custom_types import key_value_par, HostIP

EXCEPTION_TO_EXITCODE: Dict[Type[Exception], int] = {
    PermissionError: 20,
    FileNotFoundError: 30,
    SensorTagNormalizationError: 60,
    NoSuitableDataProviderError: 70,
    InsufficientDataError: 71,
    MlflowLoggingError: 80,
}

logger = logging.getLogger(__name__)


@click.group("gordo-components")
@click.version_option(version=__version__, message=__version__)
@click.option(
    "--log-level",
    type=str,
    default="INFO",
    help="Run workflow with custom log-level.",
    envvar="GORDO_LOG_LEVEL",
)
@click.pass_context
def gordo(gordo_ctx: click.Context, **ctx):
    """
    The main entry point for the CLI interface
    """
    # Set log level, defaulting to INFO
    logging.basicConfig(
        level=getattr(logging, str(gordo_ctx.params.get("log_level")).upper()),
        format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
    )

    # Set matplotlib log level to INFO to avoid noise
    logging.getLogger("matplotlib").setLevel(logging.INFO)

    gordo_ctx.obj = gordo_ctx.params


@click.command()
@click.argument("machine-config", envvar="MACHINE", type=yaml.safe_load)
@click.argument("output-dir", default="/data", envvar="OUTPUT_DIR")
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
def build(
    machine_config: dict,
    output_dir: str,
    model_register_dir: click.Path,
    print_cv_scores: bool,
    model_parameter: List[Tuple[str, Any]],
):
    """
    Build a model and deposit it into 'output_dir' given the appropriate config
    settings.

    \b
    Parameters
    ----------
    machine_config: dict
        A dict loadable by :class:`gordo_components.machine.Machine.from_config`
    output_dir: str
        Directory to save model & metadata to.
    model_register_dir: path
        Path to a directory which will index existing models and their locations, used
        for re-using old models instead of rebuilding them. If omitted then always
        rebuild
    print_cv_scores: bool
        Print cross validation scores to stdout
    model_parameter: List[Tuple[str, Any]
        List of model key-values, wheres the values will be injected into the model
        config wherever there is a jinja variable with the key.
    """
    if model_parameter and isinstance(machine_config["model"], str):
        parameters = dict(model_parameter)  # convert lib of tuples to dict
        machine_config["model"] = expand_model(machine_config["model"], parameters)

    machine: Machine = Machine.from_config(
        machine_config, project_name=machine_config["project_name"]
    )

    logger.info(f"Building, output will be at: {output_dir}")
    logger.info(f"Register dir: {model_register_dir}")

    # Convert the config into a pipeline, and back into definition to ensure
    # all default parameters are part of the config.
    logger.debug(f"Ensuring the passed model config is fully expanded.")
    machine.model = serializer.pipeline_into_definition(
        serializer.pipeline_from_definition(machine.model)
    )
    logger.info(f"Fully expanded model config: {machine.model}")

    builder = ModelBuilder(machine=machine)

    try:
        _, machine_out = builder.build(output_dir, model_register_dir)  # type: ignore

        if print_cv_scores:
            for score in get_all_score_strings(machine_out):
                print(score)
        # If enabled, configure remote logging to AzureML, otherwise logs locally
        if machine.runtime:
            try:
                logging_enabled = machine.runtime["builder"]["remote_logging"]["enable"]
            except KeyError:
                logger.debug("Remote logging doesn't appear to be enabled.")
            else:
                if logging_enabled:
                    workspace_kwargs = mlflow_utils.get_workspace_kwargs()
                    service_principal_kwargs = mlflow_utils.get_spauth_kwargs()
                    with mlflow_utils.mlflow_context(
                        machine.name,
                        builder.cache_key,
                        workspace_kwargs,
                        service_principal_kwargs,
                    ) as (mlflow_client, run_id):
                        mlflow_utils.log_metadata(mlflow_client, run_id, machine_out)

    except Exception as e:
        exit_code = EXCEPTION_TO_EXITCODE.get(e.__class__, 1)
        traceback.print_exc()
        sys.exit(exit_code)
    else:
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
    return yaml.safe_load(model_config)


def get_all_score_strings(machine):
    """Given metadata from the model builder this function returns a list of
    strings of the following format:
    {metric_name}-{tag_name}_{fold-fold-number} = {score_val}.  This computes the score for the given tag and
    cross validation split.
    {metric_name}-{tag_name}_{fold-aggregation} = {score_val}. This computes the score for the given tag and aggregates
    the score over all cross validation splits (aggregations currently used are mean, std, min and max)
    {metric_name}_{fold-fold-number} = {score_val}.  This computes the score aggregate across all tags (uses sklearn's default
    aggregation method) for a given cross validation split.
    {metric_name}_{fold-aggregation} = {score_val}.  This computes the score aggregate across all tags (uses sklearn's default
    aggregation method) and cross validation splits (aggregations currently used are mean, std, min and max).

    for katib to pick up.

    Current metric names supported are sklearn score functions: 'r2_score', 'explained_variance_score',
    'mean_squared_error' and 'mean_absolute_error'.  The underscores in such score names are replaced by '-'.

    All spaces in the tag name are also replaced by '-'.

    Parameters
    ----------
    machine : Machine
    """
    all_scores = []
    for (
        metric_name,
        scores,
    ) in machine.metadata.build_metadata.model.cross_validation.scores.items():
        metric_name = metric_name.replace(" ", "-")
        for score_name, score_val in scores.items():
            score_name = score_name.replace(" ", "-")
            all_scores.append(f"{metric_name}_{score_name}={score_val}")
    return all_scores


@click.command("run-server")
@click.option(
    "--host",
    type=HostIP(),
    help="The host to run the server on.",
    default="0.0.0.0",
    envvar="GORDO_SERVER_HOST",
    show_default=True,
)
@click.option(
    "--port",
    type=click.IntRange(1, 65535),
    help="The port to run the server on.",
    default=5555,
    envvar="GORDO_SERVER_PORT",
    show_default=True,
)
@click.option(
    "--workers",
    type=click.IntRange(1, 4),
    help="The number of worker processes for handling requests.",
    default=2,
    envvar="GORDO_SERVER_WORKERS",
    show_default=True,
)
@click.option(
    "--worker-connections",
    type=click.IntRange(1, 4000),
    help="The maximum number of simultaneous clients per worker process.",
    default=50,
    envvar="GORDO_SERVER_WORKER_CONNECTIONS",
    show_default=True,
)
@click.option(
    "--log-level",
    type=click.Choice(Logger.LOG_LEVELS.keys()),
    help="The log level for the server.",
    default="debug",
    envvar="GORDO_SERVER_LOG_LEVEL",
    show_default=True,
)
def run_server_cli(host, port, workers, worker_connections, log_level):
    """
    Run the gordo server app with Gunicorn
    """
    server.run_server(host, port, workers, worker_connections, log_level.lower())


gordo.add_command(workflow_cli)
gordo.add_command(build)
gordo.add_command(run_server_cli)
gordo.add_command(gordo_client)

if __name__ == "__main__":
    gordo()
