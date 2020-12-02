# -*- coding: utf-8 -*-

import io
from typing import Iterable, Tuple, Union

from sklearn.base import BaseEstimator

from gordo.workflow.config_elements.normalized_config import NormalizedConfig
from gordo.workflow.workflow_generator.workflow_generator import get_dict_from_yaml
from gordo.builder import ModelBuilder
from gordo.machine import Machine


def local_build(
    config_str: str,
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
    ...                 - sklearn.decomposition.PCA
    ...                 - sklearn.multioutput.MultiOutputRegressor:
    ...                     estimator: sklearn.linear_model.LinearRegression
    ...         name: crazy-sweet-name
    ... '''
    >>> models_n_metadata = local_build(config)
    >>> assert len(list(models_n_metadata)) == 1

    Returns
    -------
    Iterable[Tuple[Union[BaseEstimator, None], Machine]]
        A generator yielding tuples of models and their metadata.
    """

    config = get_dict_from_yaml(io.StringIO(config_str))
    normed = NormalizedConfig(config, project_name="local-build")
    for machine in normed.machines:
        yield ModelBuilder(machine=machine).build()
