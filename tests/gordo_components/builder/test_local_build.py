# -*- coding: utf-8 -*-

import pytest
import mock

from gordo_components.builder import local_build


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
          gordo_components.machine.model.anomaly.diff.DiffBasedAnomalyDetector:
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
          gordo_components.machine.model.anomaly.diff.DiffBasedAnomalyDetector:
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
    models_n_metadata = list(local_build(config, enable_mlflow=False))
    assert len(models_n_metadata) == 1

    model_n_metadata = models_n_metadata.pop()
    assert isinstance(model_n_metadata, tuple)
    assert len(model_n_metadata) == 2
    assert hasattr(model_n_metadata[0], "fit")
    assert isinstance(model_n_metadata[1], dict)


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
              gordo_components.machine.model.anomaly.diff.DiffBasedAnomalyDetector:
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
        list(local_build(config, enable_mlflow=False))


@mock.patch("gordo_components.builder.mlflow_utils.MlflowClient", autospec=True)
@pytest.mark.parametrize("enable_mlflow", [True, False])
def test_local_builder_mlflow(MockClient, enable_mlflow):
    """
    Test that logging is called when mlflow flag is enabled
    """

    config = """
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
        name: crazy-sweet-name1
      - dataset:
          tags:
            - SOME-TAG1
            - SOME-TAG2
          train_end_date: '2019-03-01T00:00:00+00:00'
          train_start_date: '2019-01-01T00:00:00+00:00'
          asset: asgb
          data_provider:
            type: RandomDataProvider
        name: crazy-sweet-name2
    globals:
        model:
          gordo_components.machine.model.anomaly.diff.DiffBasedAnomalyDetector:
            base_estimator:
              sklearn.pipeline.Pipeline:
                steps:
                - sklearn.decomposition.pca.PCA
                - sklearn.multioutput.MultiOutputRegressor:
                    estimator: sklearn.linear_model.base.LinearRegression
    """
    n_builds = len(list(local_build(config, enable_mlflow=enable_mlflow)))
    assert MockClient.call_count == (n_builds if enable_mlflow else 0)
