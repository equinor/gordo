# -*- coding: utf-8 -*-

import pytest

from gordo.builder import local_build
from gordo.machine import Machine


@pytest.mark.parametrize(
    "config",
    (
        """
    machines:
      - dataset:
          tags:
            - SOME-TAG1
            - SOME-TAG2
          target_tag_list:
            - SOME-TAG3
            - SOME-TAG4
          train_end_date: '2019-03-01T00:00:00+00:00'
          train_start_date: '2019-01-01T00:00:00+00:00'
          asset: asgb
          data_provider:
            type: RandomDataProvider
        metadata:
          information: Some sweet information about the model
        model:
          gordo.machine.model.anomaly.diff.DiffBasedAnomalyDetector:
            base_estimator:
              sklearn.pipeline.Pipeline:
                steps:
                - sklearn.decomposition.pca.PCA
                - sklearn.multioutput.MultiOutputRegressor:
                    estimator: sklearn.linear_model.base.LinearRegression
        name: crazy-sweet-name
    """,
        """
    machines:
      - dataset:
          tags:
            - SOME-TAG1
            - SOME-TAG2
          train_end_date: '2019-03-01T00:00:00+00:00'
          train_start_date: '2019-01-01T00:00:00+00:00'
          asset: asgb
          data_provider:
            type: RandomDataProvider
        name: crazy-sweet-name
    globals:
        model:
          gordo.machine.model.anomaly.diff.DiffBasedAnomalyDetector:
            base_estimator:
              sklearn.pipeline.Pipeline:
                steps:
                - sklearn.decomposition.pca.PCA
                - sklearn.multioutput.MultiOutputRegressor:
                    estimator: sklearn.linear_model.base.LinearRegression
    """,
    ),
)
def test_local_builder_valid_configs(config):
    models_n_metadata = list(local_build(config))
    assert len(models_n_metadata) == 1

    model_n_metadata = models_n_metadata.pop()
    assert isinstance(model_n_metadata, tuple)
    assert len(model_n_metadata) == 2

    model, machine = model_n_metadata
    assert hasattr(model, "fit")
    assert isinstance(machine, Machine)


@pytest.mark.parametrize(
    "config",
    (
        """
        machines:
          - dataset:
              tags:
                - SOME-TAG1
                - SOME-TAG2
              target_tag_list:
                - SOME-TAG3
                - SOME-TAG4
              train_end_date: '2019-03-01T00:00:00+00:00'
              train_start_date: '2019-01-01T00:00:00+00:00'
              asset: asgb
              data_provider:
                type: RandomDataProvider
            metadata:
              information: Some sweet information about the model
            ## OH NO, NO MODEL HERE, FAIL ##
            name: crazy-sweet-name
        """,
        """
        machines:
          - dataset:
              tags:
                - SOME-TAG1
                - SOME-TAG2
              train_end_date: '2019-03-01T00:00:00+00:00'
              train_start_date: '2019-01-01T00:00:00+00:00'
              asset: asgb
              data_provider:
                type: RandomDataProvider
            name: crazy-sweet-name-  ## <- Cannot end with ending hyphen
        globals:
            model:
              gordo.machine.model.anomaly.diff.DiffBasedAnomalyDetector:
                base_estimator:
                  sklearn.pipeline.Pipeline:
                    steps:
                    - sklearn.decomposition.pca.PCA
                    - sklearn.multioutput.MultiOutputRegressor:
                        estimator: sklearn.linear_model.base.LinearRegression
        """,
    ),
)
def test_local_builder_invalid_configs(config):
    with pytest.raises(Exception):
        list(local_build(config))
