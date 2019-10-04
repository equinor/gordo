# -*- coding: utf-8 -*-
import os
import dateutil.parser
import yaml
from sklearn.base import BaseEstimator

from typing import List, Optional, Dict
from tempfile import TemporaryDirectory

import pytest
import numpy as np

from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression

import gordo_components
from gordo_components.builder.build_model import (
    _save_model_for_workflow,
    provide_saved_model,
    _get_metadata,
    _determine_offset,
)
from gordo_components.builder import build_model
from gordo_components.dataset.sensor_tag import SensorTag

from gordo_components.model import models
from gordo_components.serializer import serializer


def get_random_data():
    data = {
        "type": "RandomDataset",
        "from_ts": dateutil.parser.isoparse("2017-12-25 06:00:00Z"),
        "to_ts": dateutil.parser.isoparse("2017-12-30 06:00:00Z"),
        "tag_list": [SensorTag("Tag 1", None), SensorTag("Tag 2", None)],
        "target_tag_list": [SensorTag("Tag 1", None), SensorTag("Tag 2", None)],
    }
    return data


def metadata_check(metadata, check_history):
    """Helper to verify model builder metadata creation"""
    assert "name" in metadata
    assert "model" in metadata
    assert "cross-validation" in metadata["model"]
    assert "scores" in metadata["model"]["cross-validation"]
    assert "model-offset" in metadata["model"]
    assert isinstance(metadata["model"]["model-offset"], int)

    # Scores is allowed to be an empty dict. in a case where the pipeline/transformer
    # doesn't implement a .score()
    if metadata["model"]["cross-validation"]["scores"] != dict():
        tag_list = [
            tag.name.replace(" ", "-") for tag in metadata["dataset"]["tag_list"]
        ]
        scores_list = [
            "r2-score",
            "explained-variance-score",
            "mean-squared-error",
            "mean-absolute-error",
        ]
        all_scores_list = [
            f"{score}-{tag}" for score in scores_list for tag in tag_list
        ] + scores_list
        scores_metadata = metadata["model"]["cross-validation"]["scores"]

        assert all(score in scores_metadata for score in all_scores_list)

    if check_history:
        assert "history" in metadata["model"]
        assert all(
            name in metadata["model"]["history"]
            for name in ("params", "loss", "accuracy")
        )


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
    offset = _determine_offset(model, X)
    assert offset == expected_offset


def test_output_dir(tmp_dir):
    """
    Test building of model will create subdirectories for model saving if needed.
    """
    from gordo_components.builder import build_model

    model_config = {"sklearn.decomposition.pca.PCA": {"svd_solver": "auto"}}
    data_config = get_random_data()
    output_dir = os.path.join(tmp_dir.name, "some", "sub", "directories")

    model, metadata = build_model(
        name="model-name",
        model_config=model_config,
        data_config=data_config,
        metadata={},
    )
    metadata_check(metadata, False)

    _save_model_for_workflow(model=model, metadata=metadata, output_dir=output_dir)

    # Assert the model was saved at the location
    # using gordo_components.serializer should create some subdir(s)
    # which start with 'n_step'
    dirs = [d for d in os.listdir(output_dir) if d.startswith("n_step")]
    assert (
        len(dirs) >= 1
    ), "Expected saving of model to create at least one subdir, but got {len(dirs)}"


@pytest.mark.parametrize(
    "raw_model_config",
    (
        # Without pipeline
        """
    sklearn.preprocessing.data.MinMaxScaler:
        feature_range: [-1, 1]
    """,
        # Saves history
        """
    gordo_components.model.models.KerasAutoEncoder:
        kind: feedforward_hourglass
    """,
        # With typical pipeline
        """
    sklearn.pipeline.Pipeline:
        steps:
          - sklearn.preprocessing.data.MinMaxScaler
          - sklearn.decomposition.pca.PCA:
              svd_solver: auto
    """,
        # Nested pipelilnes
        """
        sklearn.pipeline.Pipeline:
            steps:
              - sklearn.pipeline.Pipeline:
                  steps:
                    - sklearn.preprocessing.data.MinMaxScaler
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
                    - sklearn.preprocessing.data.MinMaxScaler
                    - gordo_components.model.models.KerasAutoEncoder:
                        kind: feedforward_hourglass
                        compression_factor: 0.5
                        encoding_layers: 2
                        func: tanh
                        out_func: linear
                        epochs: 3
            transformer: sklearn.preprocessing.data.MinMaxScaler
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

    model, metadata = build_model(
        name="model-name",
        model_config=model_config,
        data_config=data_config,
        metadata={},
    )
    # Check metadata, and only verify 'history' if it's a *Keras* type model
    metadata_check(metadata, "Keras" in raw_model_config)


import sklearn.compose
import sklearn.ensemble


@pytest.mark.parametrize(
    "model,expect_empty_dict",
    (
        # Model is not a GordoBase, first parameter is a pipeline and no GordoBase either
        (
            sklearn.compose.TransformedTargetRegressor(
                regressor=sklearn.pipeline.Pipeline(
                    steps=[
                        ("s1", sklearn.preprocessing.data.MinMaxScaler()),
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
                        ("s1", sklearn.preprocessing.data.MinMaxScaler()),
                        (
                            "s2",
                            gordo_components.model.models.KerasAutoEncoder(
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
                regressor=gordo_components.model.models.KerasAutoEncoder(
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
            gordo_components.model.models.KerasAutoEncoder(
                kind="feedforward_hourglass"
            ),
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

    metadata = _get_metadata(model)

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
    gordo_components.model.anomaly.diff.DiffBasedAnomalyDetector:
        scaler: sklearn.preprocessing.data.MinMaxScaler
        base_estimator:
            sklearn.compose.TransformedTargetRegressor:
                transformer: sklearn.preprocessing.data.MinMaxScaler
                regressor:
                    sklearn.pipeline.Pipeline:
                        steps:
                        - sklearn.preprocessing.data.MinMaxScaler
                        - gordo_components.model.models.KerasAutoEncoder:
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
       transformer: sklearn.preprocessing.data.MinMaxScaler
       regressor:
          sklearn.pipeline.Pipeline:
              steps:
              - sklearn.preprocessing.data.MinMaxScaler
              - gordo_components.model.models.KerasAutoEncoder:
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
      - sklearn.preprocessing.data.MinMaxScaler
      - gordo_components.model.models.KerasAutoEncoder:
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
    model, metadata = build_model(
        name="model-name",
        model_config=model_config,
        data_config=data_config,
        metadata={},
    )
    metadata_check(metadata, False)


def test_output_scores_metadata():
    data_config = get_random_data()
    raw_model_config = f"""
            gordo_components.model.anomaly.diff.DiffBasedAnomalyDetector:
                scaler: sklearn.preprocessing.data.MinMaxScaler
                base_estimator:
                    sklearn.compose.TransformedTargetRegressor:
                        transformer: sklearn.preprocessing.data.MinMaxScaler
                        regressor:
                            sklearn.pipeline.Pipeline:
                                steps:
                                - sklearn.preprocessing.data.MinMaxScaler
                                - gordo_components.model.models.KerasAutoEncoder:
                                    kind: feedforward_hourglass
                                    batch_size: 3
                                    compression_factor: 0.5
                                    encoding_layers: 1
                                    func: tanh
                                    out_func: linear
                                    epochs: 1
            """

    model_config = yaml.load(raw_model_config, Loader=yaml.FullLoader)
    model, metadata = build_model(
        name="model-name",
        model_config=model_config,
        data_config=data_config,
        metadata={},
    )
    scores_metadata = metadata["model"]["cross-validation"]["scores"]
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


def test_provide_saved_model_simple_happy_path(tmp_dir):
    """
    Test provide_saved_model with no caching
    """
    model_config = {"sklearn.decomposition.pca.PCA": {"svd_solver": "auto"}}
    data_config = get_random_data()
    output_dir = os.path.join(tmp_dir.name, "model")

    model_location = provide_saved_model(
        name="model-name",
        model_config=model_config,
        data_config=data_config,
        metadata={},
        output_dir=output_dir,
    )

    # Assert the model was saved at the location
    # using gordo_components.serializer should create some subdir(s)
    # which start with 'n_step'
    dirs = [d for d in os.listdir(model_location) if d.startswith("n_step")]
    assert (
        len(dirs) >= 1
    ), "Expected saving of model to create at least one subdir, but got {len(dirs)}"


def test_provide_saved_model_caching_handle_existing_same_dir(tmp_dir):
    """If the model exists in the model register, and the path there is the
    same as output_dir, output_dir is returned"""
    model_config = {"sklearn.decomposition.pca.PCA": {"svd_solver": "auto"}}
    data_config = get_random_data()
    output_dir = os.path.join(tmp_dir.name, "model")
    registry_dir = os.path.join(tmp_dir.name, "registry")

    model_location1 = provide_saved_model(
        name="model-name",
        model_config=model_config,
        data_config=data_config,
        metadata={},
        output_dir=output_dir,
        model_register_dir=registry_dir,
    )

    assert model_location1 == output_dir

    # Saving to same output_dir as the one saved in the registry just returns the
    # output_dir
    model_location2 = provide_saved_model(
        name="model-name",
        model_config=model_config,
        data_config=data_config,
        metadata={},
        output_dir=output_dir,
        model_register_dir=registry_dir,
    )
    assert model_location2 == output_dir


def test_provide_saved_model_caching_handle_existing_different_register(tmp_dir):
    """If the model exists in the model register, but the output_dir is not where
    the model is, the model is copied to the new location, unless the new location
    already exists. If it does then return it"""
    model_config = {"sklearn.decomposition.pca.PCA": {"svd_solver": "auto"}}
    data_config = get_random_data()
    output_dir1 = os.path.join(tmp_dir.name, "model1")
    output_dir2 = os.path.join(tmp_dir.name, "model2")

    registry_dir = os.path.join(tmp_dir.name, "registry")

    provide_saved_model(
        name="model-name",
        model_config=model_config,
        data_config=data_config,
        metadata={},
        output_dir=output_dir1,
        model_register_dir=registry_dir,
    )

    model_location2 = provide_saved_model(
        name="model-name",
        model_config=model_config,
        data_config=data_config,
        metadata={},
        output_dir=output_dir2,
        model_register_dir=registry_dir,
    )
    assert model_location2 == output_dir2

    model_location3 = provide_saved_model(
        name="model-name",
        model_config=model_config,
        data_config=data_config,
        metadata={},
        output_dir=output_dir2,
        model_register_dir=registry_dir,
    )
    assert model_location3 == output_dir2


@pytest.mark.parametrize(
    "should_be_equal,metadata,tag_list,replace_cache",
    [
        (True, None, None, False),
        (False, {"metadata": "something"}, None, False),
        (False, None, [SensorTag("extra_tag", None)], False),
        (False, None, None, True),  # replace_cache gives a new model location
    ],
)
def test_provide_saved_model_caching(
    should_be_equal: bool,
    metadata: Optional[Dict],
    tag_list: Optional[List[SensorTag]],
    replace_cache,
):
    """
    Test provide_saved_model with caching and possible cache busting if metadata,
    tag_list, or replace_cache is set.

    Builds two models and checks if their model-creation-date's are the same,
    which will be if and only if there is caching.

    Parameters
    ----------
    should_be_equal : bool
        Do we expect the two generated models to be at the same location or not? I.e. do
        we expect caching.
    metadata
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
        metadata = dict()
    with TemporaryDirectory() as tmpdir:

        model_config = {"sklearn.decomposition.pca.PCA": {"svd_solver": "auto"}}
        data_config = get_random_data()
        output_dir = os.path.join(tmpdir, "model")
        registry_dir = os.path.join(tmpdir, "registry")

        model_location = provide_saved_model(
            name="model-name",
            model_config=model_config,
            data_config=data_config,
            output_dir=output_dir,
            metadata={},
            model_register_dir=registry_dir,
        )

        if tag_list:
            data_config["tag_list"] = tag_list
        new_output_dir = os.path.join(tmpdir, "model2")
        model_location2 = provide_saved_model(
            name="model-name",
            model_config=model_config,
            data_config=data_config,
            output_dir=new_output_dir,
            metadata=metadata,
            model_register_dir=registry_dir,
            replace_cache=replace_cache,
        )

        first_metadata = serializer.load_metadata(str(model_location))
        second_metadata = serializer.load_metadata(str(model_location2))

        model1_creation_date = first_metadata["model"]["model-creation-date"]
        model2_creation_date = second_metadata["model"]["model-creation-date"]
        if should_be_equal:
            assert model1_creation_date == model2_creation_date
        else:
            assert model1_creation_date != model2_creation_date


@pytest.mark.parametrize(
    "should_be_equal,evaluation_config",
    [(True, {"cv_mode": "full_build"}), (False, {"cv_mode": "cross_val_only"})],
)
def test_model_builder_cv_scores_only(should_be_equal: bool, evaluation_config: dict):
    """
    Test checks that the model is None if cross_val_only is used as the cv_mode.
    If the default mode ('full_build') is used, the model should not be None.

    Parameters
    ----------
    should_be_equal: bool
        Refers to whether or not the cv_mode should be equal to full (default) or cross_val only.
    evaluation_config: dict
        The mode which is tested from within the evaluation_config, is either full or cross_val_only

    """

    model_config = {"sklearn.decomposition.pca.PCA": {"svd_solver": "auto"}}
    data_config = get_random_data()

    model, metadata = build_model(
        name="model-name",
        model_config=model_config,
        data_config=data_config,
        metadata={},
        evaluation_config=evaluation_config,
    )
    if should_be_equal:
        assert model is not None
    else:
        assert model is None
