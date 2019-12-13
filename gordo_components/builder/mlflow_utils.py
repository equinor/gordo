from contextlib import contextmanager
from datetime import datetime
import json
import logging
import os
import tempfile
from typing import List
from uuid import uuid4

from azureml.core import Workspace
from azureml.core.authentication import (
    InteractiveLoginAuthentication,
    ServicePrincipalAuthentication,
)
from mlflow.entities import Metric, Param
from mlflow.tracking import MlflowClient
from pytz import UTC

from gordo_components.dataset.sensor_tag import normalize_sensor_tags

logger = logging.getLogger(__name__)


class MlflowLoggingError(ValueError):
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


def get_params(d: dict, keys: List[str]) -> List[Param]:
    """
    Extract list of keys and their values from a dictionary as MLflow Params

    Parameters
    ----------
    d: dict
        Dictionary to extract key, value pairs from.
    keys: List[str]
        List of keys to extract.

    Returns
    -------
    param_list: List[Param]
        List of extracted key, value pairs stored as `mlflow.entities.Param`s.
    """
    param_list: List[Param] = list()
    for k in keys:
        if k in d:
            param_list.append(
                Param(k, d[k].isoformat() if type(d[k]) is datetime else str(d[k]))
            )
    return param_list


def get_metrics(d: dict, keys: List[str]) -> List[Metric]:
    """
    Extract list of keys and their values from a dictionary as MLflow Metrics

    Parameters
    ----------
    d: dict
        Dictionary to extract key, value pairs from.
    keys: List[str]
        List of keys to extract.

    Returns
    -------
    param_list: List[Metric]
        List of extracted key, value pairs stored as `mlflow.entities.Metric`s.
    """
    return [Metric(k, float(d[k]), epoch_now(), 0) for k in keys if k in d]


def get_batch_kwargs(metadata: dict) -> dict:
    """
    Create flat lists of MLflow logging entities from multilevel dictionary

    For more information, see the mlflow docs:
    https://www.mlflow.org/docs/latest/python_api/mlflow.tracking.html#mlflow.tracking.MlflowClient.log_batch

    Parameters
    ----------
    metadata: dict
        Multi-level dictionary with (key, value) loggables to parse.

    Returns
    -------
    batch_kwargs: dict
        Dict with `metrics` and `param_list` lists for passing to
        `mlflow.tracking.MlflowClient.log_batch`. The metrics key contains a list of
        Mlflow `Metric` instances, and the param_list key contains a list of `Param`
        instances.
    """

    param_list = get_params(metadata, ["project-name", "name"])

    dataset_keys = [
        "train_start_date",
        "train_end_date",
        "resolution",
        "filter",
        "row_filter_buffer_size",
    ]
    model_keys = ["model-creation-date", "model-builder-version", "model-offset"]
    for section, keys in zip(["dataset", "model"], [dataset_keys, model_keys]):
        param_list += get_params(metadata[section], keys)

    metric_list: List[Metric] = list()

    # Parse cross-validation metrics
    if metadata["model"].get("cross-validation", {}).get("scores", None):
        scores = metadata["model"]["cross-validation"]["scores"]
        keys = sorted(list(scores.keys()))
        subkeys = ["mean", "max", "min", "std"]

        n_folds = len(scores[keys[0]]) - len(subkeys)
        tag_list = normalize_sensor_tags(metadata["dataset"]["tag_list"])
        for k in keys:
            # Skip per tag data, produces too many params for MLflow
            if any([t.name in k for t in tag_list]):
                continue

            # Summary stats per metric
            for sk in subkeys:
                metric_list.append(
                    Metric(f"{k}_{sk}", scores[k][f"fold-{sk}"], epoch_now(), 0)
                )
            # Append value for each fold with increasing steps
            metric_list += [
                Metric(k, scores[k][f"fold-{i+1}"], epoch_now(), i)
                for i in range(n_folds)
            ]

    # Parse fit metrics
    if metadata["model"].get("history", None):
        meta_params = metadata["model"]["history"]["params"]

        metric_list += get_metrics(
            metadata["model"],
            ["data-query-duration-sec", "model-training-duration-sec"],
        )
        param_list += get_params(
            meta_params, [p for p in meta_params if p != "metrics"]
        )
        for m in meta_params["metrics"]:
            data = metadata["model"]["history"][m]
            metric_list += [
                Metric(m, float(x), timestamp=epoch_now(), step=i)
                for i, x in enumerate(data)
            ]

    return {"metrics": metric_list, "params": param_list}


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
        :func:`gordo_components.builder.mlflow_utils.get_mlflow_client`.
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
        :func:`gordo_components.builder.mlflow_utils.get_mlflow_client`
    """
    return get_kwargs_from_secret(
        "DL_SERVICE_AUTH_STR",
        ["tenant_id", "service_principal_id", "service_principal_password"],
    )


class DatetimeEncoder(json.JSONEncoder):
    """
    Encode datetime.datetime objects as strings

    Example
    -------
    >>> s = json.dumps({"now":datetime.now(tz=UTC)}, cls=DatetimeEncoder, indent=4)
    >>> s = '{"now": "2019-11-22 08:34:41.636356+"}'
    """

    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.strftime("%Y-%m-%d %H:%M:%S.%f+%z")
        else:
            return json.JSONEncoder.default(self, obj)


def send_artifacts(client: MlflowClient, run_id: str, metadata: dict):
    """
    Send artifacts

    Parameters
    ----------
    client: MlflowClient
        MLflow tracking client.
    run_id: str
        MLflow Run ID.
    metadata: dict
        Metadata produced by :func:`~gordo_components.builder.build_model.build_model`
        to extract configs from.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        for name, obj in [
            ("model-config", metadata["model"]["model-config"]),
            ("dataset-config", metadata["dataset"]),
            ("user-config", metadata["user-defined"]),
        ]:
            fp = os.path.join(tmp_dir, f"{name}.json")
            json.dump(obj, open(fp, "w"), cls=DatetimeEncoder)
        client.log_artifacts(run_id, local_dir=tmp_dir)


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
        :func:`gordo_components.builder.mlflow_utils.get_mlflow_client`.
    service_principal_kwargs: dict
        AzureML ServicePrincipalAuthentication keyword arguments. See
        :func:`gordo_components.builder.mlflow_utils.get_mlflow_client`

    Example
    -------
    >>> with tempfile.TemporaryDirectory as tmp_dir:
    ...     mlflow.set_tracking_uri(f"file:{tmp_dir}")
    ...     with mlflow_context("log_group", "unique_key", {}, {}) as (mlflow_client, run_id):
    ...         log_metadata(metadata) # doctest: +SKIP
    """
    mlflow_client = get_mlflow_client(workspace_kwargs, service_principal_kwargs)
    run_id = get_run_id(mlflow_client, experiment_name=name, model_key=model_key)

    logger.info(
        f"MLflow client configured to use {'AzureML' if workspace_kwargs else 'local backend'}"
    )

    yield mlflow_client, run_id

    mlflow_client.set_terminated(run_id)


def log_metadata(mlflow_client, run_id, metadata: dict):
    """
    Send logs to configured MLflow backend

    Parameters
    ----------
    mlflow_client: MlflowClient
        Client instance to call logging methods from.
    run_id: str
        Unique ID off MLflow Run to log to.
    metadata: dict
        Dictionary to log with MlflowClient.
    """
    # Log params and metrics
    mlflow_client.log_batch(run_id, **get_batch_kwargs(metadata))

    # Send configs as JSON artifacts
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            for name, obj in [
                ("model-config", metadata["model"]),
                ("dataset-config", metadata["dataset"]),
                (
                    "user-config",
                    metadata.get("user-defined") or metadata.get("metadata"),
                ),
            ]:
                fp = os.path.join(tmp_dir, f"{name}.json")
                json.dump(obj, open(fp, "w"), cls=DatetimeEncoder)
            mlflow_client.log_artifacts(run_id, local_dir=tmp_dir)
    # Log error without exiting the build
    # FIX: when a solution is found when logging artifacts wit azureml lib,
    # this can raise an error
    except Exception as e:
        logger.error(e)
