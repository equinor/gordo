# -*- coding: utf-8 -*-

import io
from typing import Iterable, Tuple, Union

from sklearn.base import BaseEstimator

from gordo.workflow.config_elements.normalized_config import NormalizedConfig
from gordo.workflow.workflow_generator.workflow_generator import get_dict_from_yaml
from gordo.builder import ModelBuilder
from gordo.machine import Machine


def local_build(
    config_str: str, enable_mlflow: bool = True, workspace_kwargs: dict = {}
) -> Iterable[Tuple[Union[BaseEstimator, None], Machine]]:
    """
    Build model(s) from a bare Gordo config file locally.

    This is very similar to the same steps as the normal workflow generation and subsequent
    Gordo deployment process makes. Should help developing locally,
    as well as giving a good indication that your config is valid for deployment
    with Gordo.

    Parameters
    ----------
    config_str: str
        The raw yaml config file in string format.
    enable_mlflow: bool
        Flag to enable Mlflow logging of model building results. With `enable_mlflow`
        set to `True`, passing `workspace_kwargs` an empty dict (default) will result in
        local MLflow logging, while passing a dict of keyword arguments as defined in
        :func:`~gordo.builder.mlflow_utils.get_mlflow_client` will result in
        remote logging.
    workspace_kwargs: dict
        AzureML Workspace configuration to use for remote MLFlow tracking. See
        :func:`gordo.builder.azure_utils.get_mlflow_client`.

    Examples
    --------
    >>> import numpy as np
    >>> config = '''
    ... machines:
    ...       - dataset:
    ...           tags:
    ...             - SOME-TAG1
    ...             - SOME-TAG2
    ...           target_tag_list:
    ...             - SOME-TAG3
    ...             - SOME-TAG4
    ...           train_end_date: '2019-03-01T00:00:00+00:00'
    ...           train_start_date: '2019-01-01T00:00:00+00:00'
    ...           asset: asgb
    ...           data_provider:
    ...             type: RandomDataProvider
    ...         metadata:
    ...           information: Some sweet information about the model
    ...         model:
    ...           gordo.machine.model.anomaly.diff.DiffBasedAnomalyDetector:
    ...             base_estimator:
    ...               sklearn.pipeline.Pipeline:
    ...                 steps:
    ...                 - sklearn.decomposition.pca.PCA
    ...                 - sklearn.multioutput.MultiOutputRegressor:
    ...                     estimator: sklearn.linear_model.base.LinearRegression
    ...         name: crazy-sweet-name
    ... '''
    >>> models_n_metadata = local_build(config, enable_mlflow=False)
    >>> assert len(list(models_n_metadata)) == 1

    Returns
    -------
    Iterable[Tuple[Union[BaseEstimator, None], Machine]]
        A generator yielding tuples of models and their metadata.
    """
    from gordo.builder.mlflow_utils import mlflow_context, log_machine

    config = get_dict_from_yaml(io.StringIO(config_str))
    normed = NormalizedConfig(config, project_name="local-build")
    for machine in normed.machines:
        model, machine = ModelBuilder(machine=machine).build()

        if enable_mlflow:
            # This will enforce a single interactive login and automatically
            # generate a new key for each model built
            with mlflow_context(
                name=machine.name, workspace_kwargs=workspace_kwargs
            ) as (mlflow_client, run_id):
                log_machine(mlflow_client, run_id, machine)

        yield model, machine
