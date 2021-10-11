from contextlib import contextmanager
from datetime import datetime
import json
import logging
import os
import tempfile
from typing import Dict, List, Union, Tuple
from uuid import uuid4

from azureml.core import Workspace
from azureml.core.authentication import (
    InteractiveLoginAuthentication,
    ServicePrincipalAuthentication,
)
import mlflow
from mlflow.entities import Metric, Param
from mlflow.tracking import MlflowClient
from pytz import UTC

from gordo.builder import ModelBuilder
from gordo.machine import Machine
from gordo.machine.machine import MachineEncoder
from gordo.util.utils import capture_args
from gordo_dataset.sensor_tag import extract_tag_name
from .base import BaseReporter
from .exceptions import ReporterException

logger = logging.getLogger(__name__)


class MlflowLoggingError(ReporterException):
    pass


def _validate_dict(d: dict, required_keys: List[str]):
    """
    Validate the required keys are contained in provided dictionary

    Parameters
    ----------
    d: dict
        Dictionary to validate.
    required_keys: List[str]
        Keys that must be present in provided dictionary.
    """
    if any([key not in d for key in required_keys]):
        raise MlflowLoggingError(
            f"Required keys for this dictionary include {', '.join(required_keys)}."
        )


def get_mlflow_client(
    workspace_kwargs: dict = {}, service_principal_kwargs: dict = {}
) -> MlflowClient:
    """
    Set remote tracking URI for mlflow to AzureML workspace

    Parameters
    ----------
    workspace_kwargs: dict
        AzureML Workspace configuration to use for remote MLFlow tracking. An
        empty dict will result in local logging by the MlflowClient.
        Example::

            `{
                 "subscription_id":<value>,
                 "resource_group":<value>,
                 "workspace_name":<value>
             }`
    service_principal_kwargs: dict
        AzureML ServicePrincipalAuthentication keyword arguments. An empty dict
        will result in interactive authentication.
        Example::

            `{
                 "tenant_id":<value>,
                 "service_principal_id":<value>,
                 "service_principal_password":<value>
             }`

    Returns
    -------
    client: mlflow.tracking.MlflowClient
        Client with tracking uri set to AzureML if configured.
    """
    logger.info("Creating MLflow tracking client.")

    tracking_uri = None

    # Get AzureML tracking_uri if using Azure as backend
    if workspace_kwargs:
        required_keys = ["subscription_id", "resource_group", "workspace_name"]
        _validate_dict(workspace_kwargs, required_keys)

        msg = "Configuring AzureML backend with {auth_type} authentication."
        if service_principal_kwargs:
            required_keys = [
                "tenant_id",
                "service_principal_id",
                "service_principal_password",
            ]
            _validate_dict(service_principal_kwargs, required_keys)

            logger.info(msg.format(auth_type="ServicePrincipalAuthentication"))
            workspace_kwargs["auth"] = ServicePrincipalAuthentication(
                **service_principal_kwargs
            )
        else:
            logger.info(msg.format(auth_type="interactive"))
            workspace_kwargs["auth"] = InteractiveLoginAuthentication(force=True)

        tracking_uri = Workspace(**workspace_kwargs).get_mlflow_tracking_uri()

    mlflow.set_tracking_uri(tracking_uri)

    return MlflowClient(tracking_uri)


def get_run_id(client: MlflowClient, experiment_name: str, model_key: str) -> str:
    """
    Get an existing or create a new run for the given model_key and experiment_name.

    The model key corresponds to a unique configuration of the model. The corresponding
    run must be manually stopped using the `mlflow.tracking.MlflowClient.set_terminated`
    method.

    Parameters
    ----------
    client: mlflow.tracking.MlflowClient
        Client with tracking uri set to AzureML if configured.
    experiment_name: str
        Name of experiment to log to.
    model_key: str
        Unique ID of model configuration.

    Returns
    -------
    run_id: str
        Unique ID of MLflow run to log to.
    """
    experiment = client.get_experiment_by_name(experiment_name)

    experiment_id = (
        getattr(experiment, "experiment_id")
        if experiment
        else client.create_experiment(experiment_name)
    )
    return client.create_run(experiment_id, tags={"model_key": model_key}).info.run_id


def _datetime_to_ms_since_epoch(dt: datetime) -> int:
    """
    Convert datetime to milliseconds since Unix epoch (UTC)

    Parameters
    ----------
    dt: datetime.datetime
        Timestamp to convert (can be timezone aware or naive).

    Returns
    -------
    dt: int
        Timestamp as milliseconds since Unix epoch

    Example
    -------
    >>> dt = datetime(1970, 1, 1, 0, 0)
    >>> _datetime_to_ms_since_epoch(dt)
    0
    """
    epoch = datetime.utcfromtimestamp(0).replace(tzinfo=dt.tzinfo)
    dt = dt.astimezone(UTC) if dt.tzinfo else dt.replace(tzinfo=dt.tzinfo)
    return round((dt - epoch).total_seconds() * 1000.0)


def epoch_now() -> int:
    """
    Get current timestamp in UTC as milliseconds since Unix epoch.

    Returns
    -------
    now: int
        Milliseconds since Unix epoch.
    """
    return _datetime_to_ms_since_epoch(datetime.now(tz=UTC))


def get_machine_log_items(machine: Machine) -> Tuple[List[Metric], List[Param]]:
    """
    Create flat lists of MLflow logging entities from multilevel dictionary

    For more information, see the mlflow docs:
    https://www.mlflow.org/docs/latest/python_api/mlflow.tracking.html#mlflow.tracking.MlflowClient.log_batch

    Parameters
    ----------
    machine: Machine

    Returns
    -------
    metrics: List[Metric]
        List of MLFlow Metric objects to log.
    params: List[Param]
        List of MLFlow Param objects to log.
    """

    metrics: List[Metric] = list()
    build_metadata = machine.metadata.build_metadata

    # Project/machine parameters
    keys = ["project_name", "name"]
    params = [Param(attr, getattr(machine, attr)) for attr in keys]

    # Dataset parameters
    dataset_keys = [
        "train_start_date",
        "train_end_date",
        "resolution",
        "row_filter",
        "row_filter_buffer_size",
    ]
    params.extend(Param(k, str(getattr(machine.dataset, k))) for k in dataset_keys)

    # Model parameters
    model_keys = ["model_creation_date", "model_builder_version", "model_offset"]
    params.extend(Param(k, str(getattr(build_metadata.model, k))) for k in model_keys)

    # Parse cross-validation split metadata
    splits = build_metadata.model.cross_validation.splits
    params.extend(Param(k, str(v)) for k, v in splits.items())

    # Parse cross-validation metrics

    tag_list = machine.normalize_sensor_tags(machine.dataset.tag_list)
    scores = build_metadata.model.cross_validation.scores

    keys = sorted(list(scores.keys()))
    subkeys = ["mean", "max", "min", "std"]

    n_folds = len(scores[keys[0]]) - len(subkeys)
    for k in keys:
        # Skip per tag data, produces too many params for MLflow
        if any([extract_tag_name(t) in k for t in tag_list]):
            continue

        # Summary stats per metric
        for sk in subkeys:
            metrics.append(Metric(f"{k}-{sk}", scores[k][f"fold-{sk}"], epoch_now(), 0))
        # Append value for each fold with increasing steps
        metrics.extend(
            Metric(k, scores[k][f"fold-{i+1}"], epoch_now(), i) for i in range(n_folds)
        )

    # Parse fit metrics
    try:
        history_metrics = build_metadata.model.model_meta["history"]
        meta_parms = build_metadata.model.model_meta["history"]["params"]
    except KeyError:
        logger.debug(
            "Key 'build-metadata.model.history.params' not found found in metadata."
        )
    else:
        metrics.extend(
            Metric(k, float(getattr(build_metadata.model, k)), epoch_now(), 0)
            for k in ["model_training_duration_sec"]
        )
        for m in (x for x in history_metrics if x != "params"):
            data = build_metadata.model.model_meta["history"][m]
            metrics.extend(
                Metric(m, float(x), timestamp=epoch_now(), step=i)
                for i, x in enumerate(data)
            )
        params.extend(Param(k, str(meta_parms[k])) for k in meta_parms)

    return metrics, params


def batch_log_items(
    metrics: List[Metric],
    params: List[Param],
    n_max_metrics: int = 200,
    n_max_params: int = 100,
) -> List[Dict[str, Union[Metric, Param]]]:
    """
    Split metrics, params and tags to batches that satisfy limits imposed by MLFlow and AzureML

    NOTE: The default maximum number of metrics are and parameters set here are
    those set by AzureML as per today, 18 February 2020.

    Also, there the 1mb request size is not evaluated here, as doing this
    should not be necessary and is not addressable in a succint way. MLflow
    also has a limit of 1000 log items per request, but reaching this is not
    possible with AzureML's current limit on metrics.

    Parameters
    ----------
    metrics: List[Metric]
        List of MLFlow Metric objects to log.
    params: List[Param]
        List of MLFlow Param objects to log.
    n_max_metrics:int
        Limit to number of metrics AzureML allows per batch log request payload.
    n_max_params:int
        Limit to number of params MLFlow allows per batch log request payload.

    Returns
    -------
    log_batches: List[Dict[str, Union[Metric, Param]]]
        List of MlflowClinet.log_batch keyworkd arguments, split to quatnitites
        that respect limits present for MLFlow and AzureML.
    """

    def _calc_n_batches(n: int, n_max: int):
        """
        Calculate the number of batches required to log n items with batches of n_max
        """
        return (n // n_max) + (1 if (n % n_max) else 0)

    n_batches = max(
        _calc_n_batches(len(metrics), n_max_metrics),
        _calc_n_batches(len(params), n_max_params),
    )

    i = 0
    j = 0
    log_batches = list()
    for _ in range(n_batches):
        log_batches.append(
            {
                "metrics": metrics[i : i + n_max_metrics],
                "params": params[j : j + n_max_params],
            }
        )
        i += n_max_metrics
        j += n_max_params

    return log_batches


def get_kwargs_from_secret(name: str, keys: List[str]) -> dict:
    """
    Get keyword arguments dictionary from secrets environment variable

    Parameters
    ----------
    name: str
        Name of the environment variable whose content is a colon separated
        list of secrets.

    Returns
    -------
    kwargs: dict
        Dictionary of keyword arguments parsed from environment variable.
    """
    secret_str = os.getenv(name)

    if secret_str is None:
        raise MlflowLoggingError(f"The value for env var '{name}' must not be `None`.")

    if secret_str:
        elements = secret_str.split(":")
        if len(elements) != len(keys):
            raise MlflowLoggingError(
                "`keys` len {len(keys)} must be of equal length with env var {name} elements {len(elements)}."
            )
        kwargs = {key: elements[i] for i, key in enumerate(keys)}
    else:
        kwargs = {}

    return kwargs


def get_workspace_kwargs() -> dict:
    """Get AzureML keyword arguments from environment

    The name of this environment variable is set in the Argo workflow template,
    and its value should be in the format:
    `<subscription_id>:<resource_group>:<workspace_name>`.

    Returns
    -------
    workspace_kwargs: dict
        AzureML Workspace configuration to use for remote MLFlow tracking. See
        :func:`gordo.builder.mlflow_utils.get_mlflow_client`.
    """
    return get_kwargs_from_secret(
        "AZUREML_WORKSPACE_STR", ["subscription_id", "resource_group", "workspace_name"]
    )


def get_spauth_kwargs() -> dict:
    """Get AzureML keyword arguments from environment

    The name of this environment variable is set in the Argo workflow template,
    and its value should be in the format:
    `<tenant_id>:<service_principal_id>:<service_principal_password>`

    Returns
    -------
    service_principal_kwargs: dict
        AzureML ServicePrincipalAuthentication keyword arguments. See
        :func:`gordo.builder.mlflow_utils.get_mlflow_client`
    """
    return get_kwargs_from_secret(
        "DL_SERVICE_AUTH_STR",
        ["tenant_id", "service_principal_id", "service_principal_password"],
    )


@contextmanager
def mlflow_context(
    name: str,
    model_key: str = uuid4().hex,
    workspace_kwargs: dict = {},
    service_principal_kwargs: dict = {},
):
    """
    Generate MLflow logger function with either a local or AzureML backend

    Parameters
    ----------
    name: str
        The name of the log group to log to (e.g. a model name).
    model_key: str
        Unique ID of logging run.
    workspace_kwargs: dict
        AzureML Workspace configuration to use for remote MLFlow tracking. See
        :func:`gordo.builder.mlflow_utils.get_mlflow_client`.
    service_principal_kwargs: dict
        AzureML ServicePrincipalAuthentication keyword arguments. See
        :func:`gordo.builder.mlflow_utils.get_mlflow_client`

    Example
    -------
    >>> with tempfile.TemporaryDirectory as tmp_dir:
    ...     mlflow.set_tracking_uri(f"file:{tmp_dir}")
    ...     with mlflow_context("log_group", "unique_key", {}, {}) as (mlflow_client, run_id):
    ...         log_machine(machine) # doctest: +SKIP
    """
    mlflow_client = get_mlflow_client(workspace_kwargs, service_principal_kwargs)
    run_id = get_run_id(mlflow_client, experiment_name=name, model_key=model_key)

    logger.info(
        f"MLflow client configured to use {'AzureML' if workspace_kwargs else 'local backend'}"
    )

    yield mlflow_client, run_id

    mlflow_client.set_terminated(run_id)


def log_machine(mlflow_client: MlflowClient, run_id: str, machine: Machine):
    """
    Send logs to configured MLflow backend

    Parameters
    ----------
    mlflow_client: MlflowClient
        Client instance to call logging methods from.
    run_id: str
        Unique ID off MLflow Run to log to.
    machine: Machine
        Machine to log with MlflowClient.
    """
    # Log machine metrics and params
    for batch_kwargs in batch_log_items(*get_machine_log_items(machine)):
        mlflow_client.log_batch(run_id, **batch_kwargs)

    # Send configs as JSON artifacts
    try:
        with tempfile.TemporaryDirectory(dir="./") as tmp_dir:
            fp = os.path.join(tmp_dir, f"metadata.json")
            with open(fp, "w") as fh:
                json.dump(machine.to_dict(), fh, cls=MachineEncoder)
            mlflow_client.log_artifacts(run_id=run_id, local_dir=tmp_dir)
    # Map to MlflowLoggingError for coding errors in the model builder
    except Exception as e:
        raise MlflowLoggingError(e)


class MlFlowReporter(BaseReporter):
    @capture_args
    def __init__(self, *args, **kwargs):
        pass

    def report(self, machine: Machine):

        workspace_kwargs = get_workspace_kwargs()
        service_principal_kwargs = get_spauth_kwargs()
        cache_key = ModelBuilder.calculate_cache_key(machine)

        with mlflow_context(
            machine.name, cache_key, workspace_kwargs, service_principal_kwargs
        ) as (mlflow_client, run_id):
            log_machine(mlflow_client, run_id, machine)
