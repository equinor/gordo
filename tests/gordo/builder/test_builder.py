# -*- coding: utf-8 -*-
import dateutil.parser
import os
import yaml
from typing import List, Optional, Dict, Any

import pytest
import numpy as np
import pandas as pd
import sklearn.compose
import sklearn.ensemble
from sklearn.base import BaseEstimator
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from mock import patch

import gordo
from gordo.builder import ModelBuilder
from gordo.machine.dataset.sensor_tag import SensorTag
from gordo.machine.model import models
from gordo.machine import Machine
from gordo.machine.metadata import Metadata


def get_random_data():
    data = {
        "type": "RandomDataset",
        "train_start_date": dateutil.parser.isoparse("2017-12-25 06:00:00Z"),
        "train_end_date": dateutil.parser.isoparse("2017-12-30 06:00:00Z"),
        "tag_list": [SensorTag("Tag 1", None), SensorTag("Tag 2", None)],
        "target_tag_list": [SensorTag("Tag 1", None), SensorTag("Tag 2", None)],
    }
    return data


def machine_check(machine: Machine, check_history):
    """Helper to verify model builder metadata creation"""
    assert isinstance(machine.metadata.build_metadata.model.model_offset, int)

    # Scores is allowed to be an empty dict. in a case where the pipeline/transformer
    # doesn't implement a .score()
    if machine.metadata.build_metadata.model.cross_validation.scores != dict():
        tag_list = [tag.name.replace(" ", "-") for tag in machine.dataset.tag_list]
        scores_list = [
            "r2-score",
            "explained-variance-score",
            "mean-squared-error",
            "mean-absolute-error",
        ]
        all_scores_list = [
            f"{score}-{tag}" for score in scores_list for tag in tag_list
        ] + scores_list
        scores_metadata = machine.metadata.build_metadata.model.cross_validation.scores

        assert all(score in scores_metadata for score in all_scores_list)

    if check_history:
        assert "history" in machine.metadata.build_metadata.model.model_meta
        assert all(
            name in machine.metadata.build_metadata.model.model_meta["history"]
            for name in ("params", "loss", "accuracy")
        )


@pytest.mark.parametrize("scaler", [None, "sklearn.preprocessing.MinMaxScaler"])
def test_get_metrics_dict_scaler(scaler, mock):
    mock_model = mock
    metrics_list = [sklearn.metrics.mean_squared_error]
    # make the features in y be in different scales
    y = pd.DataFrame(
        np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]) * [1, 100],
        columns=["Tag 1", "Tag 2"],
    )
    metrics_dict = ModelBuilder.build_metrics_dict(metrics_list, y, scaler=scaler)
    metric_func = metrics_dict["mean-squared-error"]

    mock_model.predict = lambda _y: _y * [0.8, 1]
    mse_feature_one_wrong = metric_func(mock_model, y, y)
    mock_model.predict = lambda _y: _y * [1, 0.8]
    mse_feature_two_wrong = metric_func(mock_model, y, y)

    if scaler:
        assert np.isclose(mse_feature_one_wrong, mse_feature_two_wrong)
    else:
        assert not np.isclose(mse_feature_one_wrong, mse_feature_two_wrong)


@pytest.mark.parametrize(
    "model,expected_offset",
    (
        (models.KerasLSTMAutoEncoder(kind="lstm_hourglass", lookback_window=10), 9),
        (models.KerasLSTMForecast(kind="lstm_symmetric", lookback_window=13), 13),
        (models.KerasAutoEncoder(kind="feedforward_hourglass"), 0),
        (MultiOutputRegressor(LinearRegression()), 0),
    ),
)
def test_determine_offset(model: BaseEstimator, expected_offset: int):
    """
    Determine the correct output difference from the model
    """
    X, y = np.random.random((100, 10)), np.random.random((100, 10))
    model.fit(X, y)
    offset = ModelBuilder._determine_offset(model, X)
    assert offset == expected_offset


def test_output_dir(tmpdir):
    """
    Test building of model will create subdirectories for model saving if needed.
    """
    model_config = {"sklearn.decomposition.pca.PCA": {"svd_solver": "auto"}}
    data_config = get_random_data()
    output_dir = os.path.join(tmpdir, "some", "sub", "directories")
    machine = Machine(
        name="model-name", dataset=data_config, model=model_config, project_name="test"
    )
    builder = ModelBuilder(machine)
    model, machine_out = builder.build()
    machine_check(machine_out, False)

    builder._save_model(model=model, machine=machine_out, output_dir=output_dir)

    # Assert the model was saved at the location
    # Should be model file, and the metadata
    assert len(os.listdir(output_dir)) == 2


@pytest.mark.parametrize(
    "raw_model_config",
    (
        # Without pipeline
        """
    sklearn.preprocessing.MinMaxScaler:
        feature_range: [-1, 1]
    """,
        # Saves history
        """
    gordo.machine.model.models.KerasAutoEncoder:
        kind: feedforward_hourglass
    """,
        # With typical pipeline
        """
    sklearn.pipeline.Pipeline:
        steps:
          - sklearn.preprocessing.MinMaxScaler
          - sklearn.decomposition.pca.PCA:
              svd_solver: auto
    """,
        # Nested pipelilnes
        """
        sklearn.pipeline.Pipeline:
            steps:
              - sklearn.pipeline.Pipeline:
                  steps:
                    - sklearn.preprocessing.MinMaxScaler
              - sklearn.pipeline.Pipeline:
                  steps:
                    - sklearn.decomposition.pca.PCA:
                        svd_solver: auto
        """,
        # Pipeline as a parameter to another estimator
        """
        sklearn.compose.TransformedTargetRegressor:
            regressor:
                sklearn.pipeline.Pipeline:
                    steps:
                    - sklearn.preprocessing.MinMaxScaler
                    - gordo.machine.model.models.KerasAutoEncoder:
                        kind: feedforward_hourglass
                        compression_factor: 0.5
                        encoding_layers: 2
                        func: tanh
                        out_func: linear
                        epochs: 3
            transformer: sklearn.preprocessing.MinMaxScaler
    """,
    ),
)
def test_builder_metadata(raw_model_config):
    """
    Ensure the builder works with various model configs and that each has
    expected/valid metadata results.
    """
    model_config = yaml.load(raw_model_config, Loader=yaml.FullLoader)
    data_config = get_random_data()
    machine = Machine(
        name="model-name", dataset=data_config, model=model_config, project_name="test"
    )
    model, machine_out = ModelBuilder(machine).build()
    # Check metadata, and only verify 'history' if it's a *Keras* type model
    machine_check(machine_out, "Keras" in raw_model_config)


@pytest.mark.parametrize(
    "model,expect_empty_dict",
    (
        # Model is not a GordoBase, first parameter is a pipeline and no GordoBase either
        (
            sklearn.compose.TransformedTargetRegressor(
                regressor=sklearn.pipeline.Pipeline(
                    steps=[
                        ("s1", sklearn.preprocessing.MinMaxScaler()),
                        ("s2", sklearn.ensemble.RandomForestRegressor()),
                    ]
                ),
                transformer=sklearn.preprocessing.MinMaxScaler(),
            ),
            True,
        ),
        # Model is not a GordoBase, first parameter is not GordoBase either.
        (
            sklearn.compose.TransformedTargetRegressor(
                regressor=sklearn.ensemble.RandomForestRegressor(),
                transformer=sklearn.preprocessing.MinMaxScaler(),
            ),
            True,
        ),
        # Model is not a GordoBase, first parameter is a pipeline with GordoBase
        (
            sklearn.compose.TransformedTargetRegressor(
                regressor=sklearn.pipeline.Pipeline(
                    steps=[
                        ("s1", sklearn.preprocessing.MinMaxScaler()),
                        (
                            "s2",
                            gordo.machine.model.models.KerasAutoEncoder(
                                kind="feedforward_hourglass"
                            ),
                        ),
                    ]
                ),
                transformer=sklearn.preprocessing.MinMaxScaler(),
            ),
            False,
        ),
        # Model is not GordoBase but the first parameter is.
        (
            sklearn.compose.TransformedTargetRegressor(
                regressor=gordo.machine.model.models.KerasAutoEncoder(
                    kind="feedforward_hourglass"
                ),
                transformer=sklearn.preprocessing.MinMaxScaler(),
            ),
            False,
        ),
        # Plain model, no GordoBase
        (sklearn.ensemble.RandomForestRegressor(), True),
        # GordoBase bare
        (
            gordo.machine.model.models.KerasAutoEncoder(kind="feedforward_hourglass"),
            False,
        ),
    ),
)
def test_get_metadata_helper(model: BaseEstimator, expect_empty_dict: bool):
    """
    Ensure the builder works with various model configs and that each has
    expected/valid metadata results.
    """

    X, y = np.random.random((1000, 4)), np.random.random((1000,))

    model.fit(X, y)

    metadata = ModelBuilder._extract_metadata_from_model(model)

    # All the metadata we've implemented so far is 'history', so we'll check that
    if not expect_empty_dict:
        assert "history" in metadata
        assert all(
            name in metadata["history"] for name in ("params", "loss", "accuracy")
        )
    else:
        assert dict() == metadata


@pytest.mark.parametrize(
    "raw_model_config",
    (
        f"""
    gordo.machine.model.anomaly.diff.DiffBasedAnomalyDetector:
        scaler: sklearn.preprocessing.MinMaxScaler
        base_estimator:
            sklearn.compose.TransformedTargetRegressor:
                transformer: sklearn.preprocessing.MinMaxScaler
                regressor:
                    sklearn.pipeline.Pipeline:
                        steps:
                        - sklearn.preprocessing.MinMaxScaler
                        - gordo.machine.model.models.KerasAutoEncoder:
                            kind: feedforward_hourglass
                            batch_size: 3
                            compression_factor: 0.5
                            encoding_layers: 1
                            func: tanh
                            out_func: linear
                            epochs: 1
        """,
        f"""
   sklearn.compose.TransformedTargetRegressor:
       transformer: sklearn.preprocessing.MinMaxScaler
       regressor:
          sklearn.pipeline.Pipeline:
              steps:
              - sklearn.preprocessing.MinMaxScaler
              - gordo.machine.model.models.KerasAutoEncoder:
                  kind: feedforward_hourglass
                  batch_size: 2
                  compression_factor: 0.5
                  encoding_layers: 1
                  func: tanh
                  out_func: linear
                  epochs: 1
                 """,
        f"""
  sklearn.pipeline.Pipeline:
      steps:
      - sklearn.preprocessing.MinMaxScaler
      - gordo.machine.model.models.KerasAutoEncoder:
          kind: feedforward_hourglass
          batch_size: 2
          compression_factor: 0.5
          encoding_layers: 1
          func: tanh
          out_func: linear
          epochs: 1
         """,
    ),
)
def test_scores_metadata(raw_model_config):
    data_config = get_random_data()
    model_config = yaml.load(raw_model_config, Loader=yaml.FullLoader)
    machine = Machine(
        dataset=data_config, model=model_config, name="model-name", project_name="test"
    )
    model, machine_out = ModelBuilder(machine).build()
    machine_check(machine_out, False)


def test_output_scores_metadata():
    data_config = get_random_data()
    raw_model_config = f"""
            gordo.machine.model.anomaly.diff.DiffBasedAnomalyDetector:
                scaler: sklearn.preprocessing.MinMaxScaler
                base_estimator:
                    sklearn.compose.TransformedTargetRegressor:
                        transformer: sklearn.preprocessing.MinMaxScaler
                        regressor:
                            sklearn.pipeline.Pipeline:
                                steps:
                                - sklearn.preprocessing.MinMaxScaler
                                - gordo.machine.model.models.KerasAutoEncoder:
                                    kind: feedforward_hourglass
                                    batch_size: 3
                                    compression_factor: 0.5
                                    encoding_layers: 1
                                    func: tanh
                                    out_func: linear
                                    epochs: 1
            """

    model_config = yaml.load(raw_model_config, Loader=yaml.FullLoader)
    machine = Machine(
        name="model-name", dataset=data_config, model=model_config, project_name="test"
    )
    model, machine_out = ModelBuilder(machine).build()
    scores_metadata = machine_out.metadata.build_metadata.model.cross_validation.scores
    assert (
        scores_metadata["explained-variance-score-Tag-1"]["fold-mean"]
        + scores_metadata["explained-variance-score-Tag-2"]["fold-mean"]
    ) / 2 == pytest.approx(scores_metadata["explained-variance-score"]["fold-mean"])

    assert (
        scores_metadata["r2-score-Tag-1"]["fold-mean"]
        + scores_metadata["r2-score-Tag-2"]["fold-mean"]
    ) / 2 == pytest.approx(scores_metadata["r2-score"]["fold-mean"])

    assert (
        scores_metadata["mean-squared-error-Tag-1"]["fold-mean"]
        + scores_metadata["mean-squared-error-Tag-2"]["fold-mean"]
    ) / 2 == pytest.approx(scores_metadata["mean-squared-error"]["fold-mean"])

    assert (
        scores_metadata["mean-absolute-error-Tag-1"]["fold-mean"]
        + scores_metadata["mean-absolute-error-Tag-2"]["fold-mean"]
    ) / 2 == pytest.approx(scores_metadata["mean-absolute-error"]["fold-mean"])


def test_provide_saved_model_simple_happy_path(tmpdir):
    """
    Test provide_saved_model with no caching
    """
    model_config = {"sklearn.decomposition.pca.PCA": {"svd_solver": "auto"}}
    data_config = get_random_data()
    output_dir = os.path.join(tmpdir, "model")
    machine = Machine(
        name="model-name", dataset=data_config, model=model_config, project_name="test"
    )
    ModelBuilder(machine).build(output_dir=output_dir)

    # Assert the model was saved at the location
    # Should be model file, and the metadata
    assert len(os.listdir(output_dir)) == 2


def test_provide_saved_model_caching_handle_existing_same_dir(tmpdir):
    """If the model exists in the model register, and the path there is the
    same as output_dir, output_dir is returned"""
    model_config = {"sklearn.decomposition.pca.PCA": {"svd_solver": "auto"}}
    data_config = get_random_data()
    output_dir = os.path.join(tmpdir, "model")
    registry_dir = os.path.join(tmpdir, "registry")
    machine = Machine(
        name="model-name", dataset=data_config, model=model_config, project_name="test"
    )
    builder = ModelBuilder(machine)
    builder.build(output_dir=output_dir, model_register_dir=registry_dir)
    assert builder.cached_model_path == output_dir

    # Saving to same output_dir as the one saved in the registry just returns the output_dir
    builder.build(output_dir=output_dir, model_register_dir=registry_dir)
    assert builder.cached_model_path == output_dir


def test_provide_saved_model_caching_handle_existing_different_register(tmpdir):
    """If the model exists in the model register, but the output_dir is not where
    the model is, the model is copied to the new location, unless the new location
    already exists. If it does then return it"""
    model_config = {"sklearn.decomposition.pca.PCA": {"svd_solver": "auto"}}
    data_config = get_random_data()
    output_dir1 = os.path.join(tmpdir, "model1")
    output_dir2 = os.path.join(tmpdir, "model2")

    registry_dir = os.path.join(tmpdir, "registry")
    machine = Machine(
        name="model-name", dataset=data_config, model=model_config, project_name="test"
    )
    builder = ModelBuilder(machine)
    builder.build(output_dir=output_dir1, model_register_dir=registry_dir)

    builder.build(output_dir=output_dir2, model_register_dir=registry_dir)
    assert builder.cached_model_path == output_dir2

    builder.build(output_dir=output_dir2, model_register_dir=registry_dir)
    assert builder.cached_model_path == output_dir2


@pytest.mark.parametrize(
    "should_be_equal,metadata,tag_list,replace_cache",
    [
        (True, None, None, False),
        (True, Metadata(user_defined={"metadata": "something"}), None, False),
        (False, Metadata(user_defined={"metadata": "something"}), None, True),
        (False, None, [SensorTag("extra_tag", None)], False),
        (False, None, None, True),  # replace_cache gives a new model location
    ],
)
def test_provide_saved_model_caching(
    should_be_equal: bool,
    metadata: Optional[Metadata],
    tag_list: Optional[List[SensorTag]],
    replace_cache,
    tmpdir,
):
    """
    Test provide_saved_model with caching and possible cache busting if tag_list, or replace_cache is set.

    Builds two models and checks if their model-creation-date's are the same,
    which will be if and only if there is caching.

    Parameters
    ----------
    should_be_equal : bool
        Do we expect the two generated models to be at the same location or not? I.e. do
        we expect caching.
    metadata: Metadata
        Optional metadata which will be used as metadata for the second model.
    tag_list
        Optional list of strings which be used as the taglist in the dataset for the
        second model.
    replace_cache: bool
        Should we force a model cache replacement?

    """

    if tag_list is None:
        tag_list = []
    if metadata is None:
        metadata = Metadata()

    model_config = {"sklearn.decomposition.pca.PCA": {"svd_solver": "auto"}}
    data_config = get_random_data()
    output_dir = os.path.join(tmpdir, "model")
    registry_dir = os.path.join(tmpdir, "registry")
    machine = Machine(
        name="model-name", dataset=data_config, model=model_config, project_name="test"
    )
    _, first_machine = ModelBuilder(machine).build(
        output_dir=output_dir, model_register_dir=registry_dir
    )

    if tag_list:
        data_config["tag_list"] = tag_list

    new_output_dir = os.path.join(tmpdir, "model2")
    _, second_machine = ModelBuilder(
        machine=Machine(
            name="model-name",
            dataset=data_config,
            model=model_config,
            metadata=metadata,
            project_name="test",
            runtime={"something": True},
        )
    ).build(
        output_dir=new_output_dir,
        model_register_dir=registry_dir,
        replace_cache=replace_cache,
    )

    model1_creation_date = (
        first_machine.metadata.build_metadata.model.model_creation_date
    )
    model2_creation_date = (
        second_machine.metadata.build_metadata.model.model_creation_date
    )
    assert "something" in second_machine.runtime

    if should_be_equal:
        assert model1_creation_date == model2_creation_date
    else:
        assert model1_creation_date != model2_creation_date

    if metadata is not None:
        assert metadata.user_defined == second_machine.metadata.user_defined


@pytest.mark.parametrize(
    "metrics_",
    (
        ["sklearn.metrics.r2_score"],
        ["r2_score"],  # string names for funcs in sklearn.metrics should also work
        None,
        ["sklearn.metrics.r2_score", "sklearn.metrics.explained_variance_score"],
    ),
)
def test_model_builder_metrics_list(metrics_: Optional[List[str]]):
    model_config = {
        "sklearn.multioutput.MultiOutputRegressor": {
            "estimator": "sklearn.linear_model.LinearRegression"
        }
    }
    data_config = get_random_data()

    evaluation_config: Dict[str, Any] = {"cv_mode": "full_build"}
    if metrics_:
        evaluation_config.update({"metrics": metrics_})

    machine = Machine(
        name="model-name",
        dataset=data_config,
        model=model_config,
        evaluation=evaluation_config,
        project_name="test",
    )
    _model, machine = ModelBuilder(machine).build()

    expected_metrics = metrics_ or [
        "sklearn.metrics.explained_variance_score",
        "sklearn.metrics.r2_score",
        "sklearn.metrics.mean_squared_error",
        "sklearn.metrics.mean_absolute_error",
    ]

    assert all(
        metric.split(".")[-1].replace("_", "-")
        in machine.metadata.build_metadata.model.cross_validation.scores
        for metric in expected_metrics
    )


def test_metrics_from_list():
    """
    Check getting functions from a list of metric names
    """
    default = ModelBuilder.metrics_from_list()
    assert default == [
        metrics.explained_variance_score,
        metrics.r2_score,
        metrics.mean_squared_error,
        metrics.mean_absolute_error,
    ]

    specifics = ModelBuilder.metrics_from_list(
        ["sklearn.metrics.adjusted_mutual_info_score", "sklearn.metrics.r2_score"]
    )
    assert specifics == [metrics.adjusted_mutual_info_score, metrics.r2_score]


@pytest.mark.parametrize("seed", (None, 1234))
@pytest.mark.parametrize(
    "model_config",
    (
        {
            "sklearn.multioutput.MultiOutputRegressor": {
                "estimator": "sklearn.ensemble.forest.RandomForestRegressor"
            }
        },
        {
            "gordo.machine.model.models.KerasAutoEncoder": {
                "kind": "feedforward_hourglass"
            }
        },
    ),
)
def test_setting_seed(seed, model_config):
    """
    Test that we can set the seed and get same results.
    """

    data_config = get_random_data()
    evaluation_config = {"cv_mode": "full_build", "seed": seed}

    # Training two instances, without a seed should result in different scores,
    # while doing it with a seed should result in the same scores.
    machine = Machine(
        name="model-name",
        dataset=data_config,
        model=model_config,
        evaluation=evaluation_config,
        project_name="test",
    )
    _model, machine1 = ModelBuilder(machine).build()
    _model, machine2 = ModelBuilder(machine).build()

    df1 = pd.DataFrame.from_dict(
        machine1.metadata.build_metadata.model.cross_validation.scores
    )
    df2 = pd.DataFrame.from_dict(
        machine2.metadata.build_metadata.model.cross_validation.scores
    )

    # Equality depends on the seed being set.
    if seed:
        assert df1.equals(df2)
    else:
        assert not df1.equals(df2)


@pytest.mark.parametrize(
    "cv",
    (
        {"sklearn.model_selection.TimeSeriesSplit": {"n_splits": 5}},
        {
            "sklearn.model_selection.TimeSeriesSplit": {
                "n_splits": 5,
                "max_train_size": 10,
            }
        },
        {"sklearn.model_selection.ShuffleSplit": {"n_splits": 5}},
        None,
    ),
)
@patch("gordo.serializer.from_definition")
def test_n_splits_from_config(mocked_pipeline_from_definition, cv):
    """
    Test that we can set arbitrary splitters and parameters in the config file which is called by the serializer.
    """
    data_config = get_random_data()
    evaluation_config = {"cv_mode": "full_build"}
    if cv:
        evaluation_config["cv"] = cv

    model_config = {
        "sklearn.multioutput.MultiOutputRegressor": {
            "estimator": "sklearn.ensemble.forest.RandomForestRegressor"
        }
    }

    machine = Machine(
        name="model-name",
        dataset=data_config,
        model=model_config,
        evaluation=evaluation_config,
        project_name="test",
    )

    ModelBuilder(machine).build()

    if cv:
        mocked_pipeline_from_definition.assert_called_with(cv)
    else:
        mocked_pipeline_from_definition.assert_called_with(
            {"sklearn.model_selection.TimeSeriesSplit": {"n_splits": 3}}
        )


@patch("gordo.machine.Machine.report")
def test_builder_calls_machine_report(mocked_report_method, metadata):
    """
    When building a machine, the Modelbuilder.build should call Machine.report()
    so that it can run any reporters in the Machine's runtime.
    """
    machine = Machine(**metadata)
    ModelBuilder(machine).build()
    assert mocked_report_method.called_once()
