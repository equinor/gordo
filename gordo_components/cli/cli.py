# -*- coding: utf-8 -*-

"""
CLI interfaces
"""

import logging
from gunicorn.glogging import Logger

import jinja2
import yaml
import click

from gordo_components.builder.build_model import (
    provide_saved_model,
    check_cache,
    calculate_model_key,
    build_model,
)
from gordo_components.serializer import (
    load_metadata,
    pipeline_into_definition,
    pipeline_from_definition,
)
from gordo_components.server import server
from gordo_components import watchman, __version__
from gordo_components.cli.workflow_generator import workflow_cli
from gordo_components.cli.client import client as gordo_client
from gordo_components.cli.custom_types import key_value_par, DataProviderParam, HostIP
from gordo_components.dataset.sensor_tag import normalize_sensor_tags

import dateutil.parser

logger = logging.getLogger(__name__)


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
@click.option(
    "--data-provider",
    type=DataProviderParam(),
    envvar="DATA_PROVIDER",
    default='{"type": "DataLakeProvider"}',
    help="DataProvider dict encoded as json. Must contain a 'type' key with the name of"
    " a DataProvider as value.",
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
    "--evaluation-config",
    envvar="EVALUATION_CONFIG",
    default='{"cv_mode": "full_build"}',
    type=yaml.safe_load,
)
def build(
    name,
    output_dir,
    model_config,
    data_config,
    data_provider,
    metadata,
    model_register_dir,
    print_cv_scores,
    model_parameter,
    evaluation_config,
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
    data_provider: str
        A quoted data provider configuration in  JSON/YAML format.
        Should also contain key 'type' which references the data provider to use.

        Example::

          '{"type": "DataLakeProvider", "storename" : "example_store"}'

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

    evaluation_config: dict
        Dict of parameters which are exposed to build_model.
            - cv_mode: str
                String which enables three different modes, represented as a key value in evaluation_config:
                * cross_val_only: Only perform cross validation
                * build_only: Skip cross validation and only build the model
                * full_build: Cross validation and full build of the model, default value
                Example::

                    {"cv_mode": "cross_val_only"}
    """

    data_config["tag_list"] = data_config.pop("tags")

    data_config["from_ts"] = dateutil.parser.isoparse(
        data_config.pop("train_start_date")
    )

    data_config["to_ts"] = dateutil.parser.isoparse(data_config.pop("train_end_date"))

    # Set default data provider for data config
    data_config["data_provider"] = data_provider
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

    if evaluation_config["cv_mode"] == "cross_val_only":

        cache_model_location = None
        if model_register_dir is not None:
            cache_key = calculate_model_key(
                name, model_config, data_config, evaluation_config, metadata=metadata
            )
            cache_model_location = check_cache(model_register_dir, cache_key)

        if cache_model_location:
            metadata = load_metadata(cache_model_location)
        else:
            _, metadata = build_model(
                name, model_config, data_config, metadata, evaluation_config
            )

    else:
        model_location = provide_saved_model(
            name,
            model_config,
            data_config,
            metadata,
            output_dir,
            model_register_dir,
            evaluation_config=evaluation_config,
        )
        metadata = load_metadata(model_location)

    # If the model is cached but without CV scores then we force a rebuild. We do this
    # by deleting the entry in the cache and then rerun `provide_saved_model`
    # (leaving the old model laying around)
    if print_cv_scores:
        retrieved_metadata = metadata
        all_scores = get_all_score_strings(retrieved_metadata)
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
                evaluation_config=evaluation_config,
            )
            saved_metadata = load_metadata(model_location)
            all_scores = get_all_score_strings(saved_metadata)

        for score in all_scores:
            print(score)

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
    ...             "scores": {"explained variance tag 0": {"fold_1": 0, "fold_2": 0.9, "fold_3": 0.1,"min": 0.1, "max": 0.9, "mean": 1/3},
    ...                        "explained variance tag 1": {"fold_1": 0.2, "fold_2": 0.3, "fold_3": 0.6, "min": 0.2, "max": 0.6, "mean": 0.3666666666666667},
    ...                        "explained variance" : {"min": 0.1, "max": 0.6, "mean": 0.3499999999999999},
    ...                        "r2 score tag 0" : {"fold_1": 0.8, "fold_2": 0.5, "fold_3": 0.6, "min": 0.5, "max": 0.8, "mean": 0.6333333333333333},
    ...                        "r2 score tag 1" : {"fold_1": 0.4, "fold_2": 0.3, "fold_3": 0.5, "min": 0.3, "max": 0.5, "mean": 0.39999999999999997},
    ...                        "r2 score"  : {"min": 0.4,"max": 0.6, "mean": 0.5166666666666666}
    ...                          }
    ...     }
    ...   }
    ... }
    ... )
    >>> len(score_strings)
    30
    >>> score_strings
    ['explained-variance-tag-0_fold_1=0', 'explained-variance-tag-0_fold_2=0.9', 'explained-variance-tag-0_fold_3=0.1', 'explained-variance-tag-0_min=0.1', 'explained-variance-tag-0_max=0.9', 'explained-variance-tag-0_mean=0.3333333333333333', 'explained-variance-tag-1_fold_1=0.2', 'explained-variance-tag-1_fold_2=0.3', 'explained-variance-tag-1_fold_3=0.6', 'explained-variance-tag-1_min=0.2', 'explained-variance-tag-1_max=0.6', 'explained-variance-tag-1_mean=0.3666666666666667', 'explained-variance_min=0.1', 'explained-variance_max=0.6', 'explained-variance_mean=0.3499999999999999', 'r2-score-tag-0_fold_1=0.8', 'r2-score-tag-0_fold_2=0.5', 'r2-score-tag-0_fold_3=0.6', 'r2-score-tag-0_min=0.5', 'r2-score-tag-0_max=0.8', 'r2-score-tag-0_mean=0.6333333333333333', 'r2-score-tag-1_fold_1=0.4', 'r2-score-tag-1_fold_2=0.3', 'r2-score-tag-1_fold_3=0.5', 'r2-score-tag-1_min=0.3', 'r2-score-tag-1_max=0.5', 'r2-score-tag-1_mean=0.39999999999999997', 'r2-score_min=0.4', 'r2-score_max=0.6', 'r2-score_mean=0.5166666666666666']


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


gordo.add_command(workflow_cli)
gordo.add_command(build)
gordo.add_command(run_server_cli)
gordo.add_command(run_watchman_cli)
gordo.add_command(gordo_client)

if __name__ == "__main__":
    gordo()
