# -*- coding: utf-8 -*-

"""
CLI interfaces
"""

import logging
import ast
import os
from ast import literal_eval
from dateutil import parser

import click
from gordo_components.builder import build_model
from gordo_components.builder.build_model import _save_model_for_workflow
from gordo_components.server import server
from gordo_components.dataset.datasets import TimeSeriesDataset
from gordo_components.data_provider.providers import DataLakeProvider
from gordo_components import watchman


# Set log level, defaulting to DEBUG
log_level = os.getenv("LOG_LEVEL", "DEBUG").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
)
logger = logging.getLogger(__name__)


@click.group("gordo-components")
def gordo():
    """
    The main entry point for the CLI interface
    """
    pass


@click.group("build")
@click.option("--metadata", envvar="METADATA", default="{}", type=literal_eval)
@click.pass_context
def build(ctx: click.Context, metadata: dict):
    ctx.obj = dict()
    ctx.obj["metadata"] = metadata
    pass


def date_string_or_literal(val):
    """
    Given a value, attempt to parse it as iso datetime object -> literal -> string
    in that order.
    """
    logger.info(f'Parsing "{val}" of type {type(val)}')
    try:
        return parser.isoparse(val)
    except ValueError:
        try:
            return literal_eval(str(val))
        except (SyntaxError, ValueError):
            return val


@click.command("timeseries")
@click.argument("output-dir", default="/data", envvar="OUTPUT_DIR")
@click.argument("model-type", default="KerasAutoEncoder")
@click.argument("from_ts", type=parser.isoparse)
@click.argument("to_ts", type=parser.isoparse)
@click.argument("tag_list", type=list)
@click.option('--resoluiton', type=str, help="Pandas frequency string, ie. '2T'")
@click.option(
    "--model-kwarg",
    type=(str, date_string_or_literal),
    multiple=True,
    help="Set a specific model kwargs, ie. --model-kwarg n_epoch 5",
)
@click.pass_context
def timeseries(ctx: click.Context, output_dir, model_type, from_ts, to_ts, resolution, tag_list, model_kwarg):
    """
    Build a model and deposit it into 'output_dir' given the appropriate config
    settings.

    \b
    Parameters
    ----------
    output_dir: str - Directory to save model & metadata to.

    TODO: Fill the remainer of this in.
    """

    logger.info(f"Got this context: {ctx.obj}")

    # Create the model config
    model_config = {k: v for k, v in model_kwarg}
    model_config.update({"type": model_type})

    # Build the timeseries dataset
    provider = DataLakeProvider({"storename": "dataplatformdlsprod"}, from_ts=from_ts, to_ts=to_ts, resolution=resolution, tag_list=tag_list)
    dataset = TimeSeriesDataset(from_ts=from_ts, to_ts=to_ts, resolution=resolution, data_provider=provider)

    # Create metadata
    metadata = ctx.obj.get('metadata', {})

    logger.info(f"Building, output will be at: {output_dir}")
    logger.info(f"Model config: {model_config}")
    logger.info(f"Data config: {data_config}")
    logger.info(f"Metadata: {metadata}")

    model, metadata = build_model(
        model_config=model_config, data_config=dataset, metadata=metadata
    )

    logger.debug(f"Saving model to output dir: {output_dir}")
    _save_model_for_workflow(model=model, metadata=metadata, output_dir=output_dir)

    logger.info(f"Successfully built model, and deposited at {output_dir}")
    return 0


@click.command("run-server")
@click.option("--host", type=str, help="The host to run the server on.")
@click.option("--port", type=int, help="The port to run the server on.")
def run_server_cli(host, port):
    server.run_server(host, port)


@click.command("run-watchman")
@click.argument("project-name", envvar="PROJECT_NAME", type=str)
@click.argument("target-names", envvar="TARGET_NAMES", type=ast.literal_eval)
@click.option(
    "--host", type=str, help="The host to run the server on.", default="0.0.0.0"
)
@click.option("--port", type=int, help="The port to run the server on.", default=5555)
@click.option("--debug", type=bool, help="Run in debug mode", default=False)
def run_watchman_cli(project_name, target_names, host, port, debug):
    """
    Start the Gordo Watchman server for this project. Which is responsible
    for dynamically comparing expected URLs derived from a project config fle
    against those actually deployed to determine and report their health.

    \b
    Must have the following environment variables set:
        PROJECT_NAME: project_name for the config file
        TARGET_NAMES: A list of non-sanitized machine / target names
    """
    watchman.server.run_server(host, port, debug, project_name, target_names)


build.add_command(timeseries)

gordo.add_command(build)
gordo.add_command(run_server_cli)
gordo.add_command(run_watchman_cli)

if __name__ == "__main__":
    gordo()
