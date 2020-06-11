# -*- coding: utf-8 -*-

from datetime import timedelta

import pytest
import numpy as np
import pandas as pd
import yaml

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
from sklearn.model_selection import TimeSeriesSplit

from gordo import serializer
from gordo.machine.model import utils as model_utils
from gordo.machine.model.anomaly import DiffBasedAnomalyDetector
from gordo.machine.model.anomaly.base import AnomalyDetectorBase


@pytest.mark.parametrize("scaler", (MinMaxScaler(), RobustScaler()))
@pytest.mark.parametrize(
    "index", (range(10), pd.date_range("2019-01-01", "2019-01-30", periods=10))
)
@pytest.mark.parametrize("lookback", (0, 5))
@pytest.mark.parametrize("with_thresholds", (True, False))
def test_diff_detector(scaler, index, lookback, with_thresholds: bool):
    """
    Test the functionality of the DiffBasedAnomalyDetector
    """

    # Some dataset.
    X, y = (
        pd.DataFrame(np.random.random((10, 3))),
        pd.DataFrame(np.random.random((10, 3))),
    )

    base_estimator = MultiOutputRegressor(estimator=LinearRegression())
    model = DiffBasedAnomalyDetector(
        base_estimator=base_estimator, scaler=scaler, require_thresholds=False
    )

    assert isinstance(model, AnomalyDetectorBase)

    assert model.get_params() == dict(base_estimator=base_estimator, scaler=scaler)

    if with_thresholds:
        model.cross_validate(X=X, y=y)

    model.fit(X, y)

    output: np.ndarray = model.predict(X)
    base_df = model_utils.make_base_dataframe(
        tags=["A", "B", "C"], model_input=X, model_output=output, index=index
    )

    # Base prediction dataframe has none of these columns
    assert not any(
        col in base_df.columns
        for col in (
            "total-anomaly-scaled",
            "total-anomaly-unscaled",
            "tag-anomaly-scaled",
            "tag-anomaly-unscaled",
        )
    )

    # Apply the anomaly detection logic on the base prediction df
    anomaly_df = model.anomaly(X, y, timedelta(days=1))

    # Should have these added error calculated columns now.
    assert all(
        col in anomaly_df.columns
        for col in (
            "total-anomaly-scaled",
            "total-anomaly-unscaled",
            "tag-anomaly-scaled",
            "tag-anomaly-unscaled",
        )
    )

    # Verify calculation for unscaled data
    feature_error_unscaled = np.abs(base_df["model-output"].values - y.values)
    total_anomaly_unscaled = np.square(feature_error_unscaled).mean(axis=1)
    assert np.allclose(
        feature_error_unscaled, anomaly_df["tag-anomaly-unscaled"].values
    )
    assert np.allclose(
        total_anomaly_unscaled, anomaly_df["total-anomaly-unscaled"].values
    )

    # Verify calculations for scaled data
    feature_error_scaled = np.abs(
        scaler.transform(base_df["model-output"].values) - scaler.transform(y)
    )
    total_anomaly_scaled = np.square(feature_error_scaled).mean(axis=1)
    assert np.allclose(feature_error_scaled, anomaly_df["tag-anomaly-scaled"].values)
    assert np.allclose(total_anomaly_scaled, anomaly_df["total-anomaly-scaled"].values)

    if with_thresholds:
        assert "anomaly-confidence" in anomaly_df.columns
        assert "total-anomaly-confidence" in anomaly_df.columns
    else:
        assert "anomaly-confidence" not in anomaly_df.columns
        assert "total-anomaly-confidence" not in anomaly_df.columns


@pytest.mark.parametrize("scaler", (MinMaxScaler(), RobustScaler()))
@pytest.mark.parametrize("len_x_y", (100, 144, 1440))
@pytest.mark.parametrize("time_index", (True, False))
@pytest.mark.parametrize("lookback", (0, 5))
@pytest.mark.parametrize("with_thresholds", (True, False))
def test_diff_detector_with_window(
    scaler, len_x_y: int, time_index: bool, lookback: int, with_thresholds: bool
):
    """
    Test the functionality of the DiffBasedAnomalyDetector
    """

    # Some dataset.
    X, y = (
        pd.DataFrame(np.random.random((len_x_y, 3))),
        pd.DataFrame(np.random.random((len_x_y, 3))),
    )
    tags = ["A", "B", "C"]
    if time_index:
        index = pd.date_range("2019-01-01", "2019-01-11", periods=len_x_y)
    else:
        index = range(len_x_y)

    base_estimator = MultiOutputRegressor(estimator=LinearRegression())
    model = DiffBasedAnomalyDetector(
        base_estimator=base_estimator,
        scaler=scaler,
        require_thresholds=False,
        window=144,
    )

    assert isinstance(model, AnomalyDetectorBase)

    assert model.get_params() == dict(
        base_estimator=base_estimator, scaler=scaler, window=144
    )

    if with_thresholds:
        model.cross_validate(X=X, y=y)

    model.fit(X, y)

    output: np.ndarray = model.predict(X)
    base_df = model_utils.make_base_dataframe(
        tags=tags, model_input=X, model_output=output, index=index
    )
    # Base prediction dataframe has none of these columns
    assert not any(
        col in base_df.columns
        for col in (
            "total-anomaly-scaled",
            "total-anomaly-unscaled",
            "tag-anomaly-scaled",
            "tag-anomaly-unscaled",
            "smooth-total-anomaly-scaled",
            "smooth-total-anomaly-unscaled",
            "smooth-tag-anomaly-scaled",
            "smooth-tag-anomaly-unscaled",
        )
    )

    # Apply the anomaly detection logic on the base prediction df
    anomaly_df = model.anomaly(X, y)

    # Should have these added error calculated columns now.
    assert all(
        col in anomaly_df.columns
        for col in (
            "total-anomaly-scaled",
            "total-anomaly-unscaled",
            "tag-anomaly-scaled",
            "tag-anomaly-unscaled",
            "smooth-total-anomaly-scaled",
            "smooth-total-anomaly-unscaled",
            "smooth-tag-anomaly-scaled",
            "smooth-tag-anomaly-unscaled",
        )
    )

    # Verify calculation for unscaled data
    feature_error_unscaled = pd.DataFrame(
        data=np.abs(base_df["model-output"].to_numpy() - y.to_numpy()),
        index=index,
        columns=tags,
    )
    total_anomaly_unscaled = pd.Series(
        data=np.square(feature_error_unscaled).mean(axis=1)
    )
    assert np.allclose(
        feature_error_unscaled.to_numpy(), anomaly_df["tag-anomaly-unscaled"].to_numpy()
    )
    assert np.allclose(
        total_anomaly_unscaled.to_numpy(),
        anomaly_df["total-anomaly-unscaled"].to_numpy(),
    )

    smooth_feature_error_unscaled = (
        feature_error_unscaled.rolling(model.window).median().dropna()
    )
    smooth_total_anomaly_unscaled = (
        total_anomaly_unscaled.rolling(model.window).median().dropna()
    )
    assert np.allclose(
        smooth_feature_error_unscaled.to_numpy(),
        anomaly_df["smooth-tag-anomaly-unscaled"].dropna().to_numpy(),
    )
    assert np.allclose(
        smooth_total_anomaly_unscaled.to_numpy(),
        anomaly_df["smooth-total-anomaly-unscaled"].dropna().to_numpy(),
    )

    # Verify calculations for scaled data
    feature_error_scaled = pd.DataFrame(
        data=np.abs(
            scaler.transform(base_df["model-output"].to_numpy()) - scaler.transform(y)
        ),
        index=index,
        columns=tags,
    )
    total_anomaly_scaled = pd.Series(data=np.square(feature_error_scaled).mean(axis=1))
    assert np.allclose(
        feature_error_scaled.to_numpy(), anomaly_df["tag-anomaly-scaled"].to_numpy()
    )
    assert np.allclose(
        total_anomaly_scaled, anomaly_df["total-anomaly-scaled"].to_numpy()
    )

    smooth_feature_error_scaled = (
        feature_error_scaled.rolling(model.window).median().dropna()
    )
    smooth_total_anomaly_scaled = (
        total_anomaly_scaled.rolling(model.window).median().dropna()
    )
    assert np.allclose(
        smooth_feature_error_scaled.to_numpy(),
        anomaly_df["smooth-tag-anomaly-scaled"].dropna().to_numpy(),
    )
    assert np.allclose(
        smooth_total_anomaly_scaled.to_numpy(),
        anomaly_df["smooth-total-anomaly-scaled"].dropna().to_numpy(),
    )

    # Check number of NA's is consistent with window size
    if len_x_y >= model.window:
        assert (
            anomaly_df["smooth-tag-anomaly-scaled"].isna().sum().sum()
            == (model.window - 1) * anomaly_df["smooth-tag-anomaly-scaled"].shape[1]
        )
        assert (
            anomaly_df["smooth-total-anomaly-scaled"].isna().sum() == model.window - 1
        )

    if with_thresholds:
        assert "anomaly-confidence" in anomaly_df.columns
        assert "total-anomaly-confidence" in anomaly_df.columns
    else:
        assert "anomaly-confidence" not in anomaly_df.columns
        assert "total-anomaly-confidence" not in anomaly_df.columns


@pytest.mark.parametrize(
    "config",
    (
        """
        gordo.machine.model.anomaly.diff.DiffBasedAnomalyDetector:
            base_estimator:
                sklearn.pipeline.Pipeline:
                    steps:
                        - sklearn.preprocessing.MinMaxScaler
                        - gordo.machine.model.models.KerasAutoEncoder:
                            kind: feedforward_hourglass
                    memory:
    """,
        """
        gordo.machine.model.anomaly.diff.DiffBasedAnomalyDetector:
            base_estimator:
                gordo.machine.model.models.KerasAutoEncoder:
                    kind: feedforward_hourglass
    """,
    ),
)
def test_diff_detector_serializability(config):
    """
    Should play well with the gordo serializer
    """
    config = yaml.safe_load(config)

    model = serializer.from_definition(config)
    serializer.into_definition(model)
    serialized_bytes = serializer.dumps(model)
    serializer.loads(serialized_bytes)


@pytest.mark.parametrize("n_features_y", range(1, 3))
@pytest.mark.parametrize("n_features_x", range(1, 3))
def test_diff_detector_threshold(n_features_y: int, n_features_x: int):
    """
    Basic construction logic of thresholds_ attribute in the
    DiffBasedAnomalyDetector
    """
    X = np.random.random((100, n_features_x))
    y = np.random.random((100, n_features_y))

    model = DiffBasedAnomalyDetector(
        base_estimator=MultiOutputRegressor(estimator=LinearRegression())
    )

    # Model has own implementation of cross_validate
    assert hasattr(model, "cross_validate")

    # When initialized it should not have a threshold calculated.
    assert not hasattr(model, "feature_thresholds_")
    assert not hasattr(model, "aggregate_threshold_")
    assert not hasattr(model, "feature_thresholds_per_fold_")
    assert not hasattr(model, "aggregate_thresholds_per_fold_")

    model.fit(X, y)

    # Until it has done cross validation, it has no threshold.
    assert not hasattr(model, "feature_thresholds_")
    assert not hasattr(model, "aggregate_threshold_")
    assert not hasattr(model, "feature_thresholds_per_fold_")
    assert not hasattr(model, "aggregate_thresholds_per_fold_")

    # Calling cross validate should set the threshold for it.
    model.cross_validate(X=X, y=y)

    # Now we have calculated thresholds based on cross validation folds
    assert hasattr(model, "feature_thresholds_")
    assert hasattr(model, "aggregate_threshold_")
    assert hasattr(model, "feature_thresholds_per_fold_")
    assert hasattr(model, "aggregate_thresholds_per_fold_")
    assert isinstance(model.feature_thresholds_, pd.Series)
    assert len(model.feature_thresholds_) == y.shape[1]
    assert all(model.feature_thresholds_.notna())
    assert isinstance(model.feature_thresholds_per_fold_, pd.DataFrame)
    assert isinstance(model.aggregate_thresholds_per_fold_, dict)


@pytest.mark.parametrize("n_features_y", range(1, 3))
@pytest.mark.parametrize("n_features_x", range(1, 3))
@pytest.mark.parametrize("len_x_y", [100, 144, 1440])
def test_diff_detector_threshold_with_window(
    n_features_y: int, n_features_x: int, len_x_y: int
):
    """
    Basic construction logic of thresholds_ attribute in the
    DiffBasedAnomalyDetector
    """
    X = np.random.random((len_x_y, n_features_x))
    y = np.random.random((len_x_y, n_features_y))

    model = DiffBasedAnomalyDetector(
        base_estimator=MultiOutputRegressor(estimator=LinearRegression()), window=144
    )

    # Model has own implementation of cross_validate
    assert hasattr(model, "cross_validate")

    # When initialized it should not have a threshold calculated.
    assert not hasattr(model, "feature_thresholds_")
    assert not hasattr(model, "aggregate_threshold_")
    assert not hasattr(model, "feature_thresholds_per_fold_")
    assert not hasattr(model, "aggregate_thresholds_per_fold_")
    assert not hasattr(model, "smooth_feature_thresholds_")
    assert not hasattr(model, "smooth_aggregate_threshold_")
    assert not hasattr(model, "smooth_feature_thresholds_per_fold_")
    assert not hasattr(model, "smooth_aggregate_thresholds_per_fold_")

    model.fit(X, y)

    # Until it has done cross validation, it has no threshold.
    assert not hasattr(model, "feature_thresholds_")
    assert not hasattr(model, "aggregate_threshold_")
    assert not hasattr(model, "feature_thresholds_per_fold_")
    assert not hasattr(model, "aggregate_thresholds_per_fold_")
    assert not hasattr(model, "smooth_feature_thresholds_")
    assert not hasattr(model, "smooth_aggregate_threshold_")
    assert not hasattr(model, "smooth_feature_thresholds_per_fold_")
    assert not hasattr(model, "smooth_aggregate_thresholds_per_fold_")

    # Calling cross validate should set the threshold for it.
    model.cross_validate(X=X, y=y)

    # Now we have calculated thresholds based on cross validation folds
    assert hasattr(model, "feature_thresholds_")
    assert hasattr(model, "aggregate_threshold_")
    assert hasattr(model, "feature_thresholds_per_fold_")
    assert hasattr(model, "aggregate_thresholds_per_fold_")
    assert isinstance(model.feature_thresholds_, pd.Series)
    assert len(model.feature_thresholds_) == y.shape[1]
    assert all(model.feature_thresholds_.notna())
    assert isinstance(model.feature_thresholds_per_fold_, pd.DataFrame)
    assert isinstance(model.aggregate_thresholds_per_fold_, dict)

    assert hasattr(model, "smooth_feature_thresholds_")
    assert hasattr(model, "smooth_aggregate_threshold_")
    assert hasattr(model, "smooth_feature_thresholds_per_fold_")
    assert hasattr(model, "smooth_aggregate_thresholds_per_fold_")
    assert isinstance(model.smooth_feature_thresholds_, pd.Series)
    assert len(model.smooth_feature_thresholds_) == y.shape[1]
    if len_x_y <= model.window:
        assert all(model.smooth_feature_thresholds_.isna())
    else:
        assert all(model.smooth_feature_thresholds_.notna())
    assert isinstance(model.smooth_feature_thresholds_per_fold_, pd.DataFrame)
    assert isinstance(model.smooth_aggregate_thresholds_per_fold_, dict)


@pytest.mark.parametrize("return_estimator", (True, False))
def test_diff_detector_cross_validate(return_estimator: bool):
    """
    DiffBasedAnomalyDetector.cross_validate implementation should be the
    same as sklearn.model_selection.cross_validate if called the same.

    And it always will update `return_estimator` to True, as it requires
    the intermediate models to calculate the thresholds
    """
    X = np.random.random((100, 10))
    y = np.random.random((100, 1))

    model = DiffBasedAnomalyDetector(base_estimator=LinearRegression())

    cv = TimeSeriesSplit(n_splits=3)
    cv_results_da = model.cross_validate(
        X=X, y=y, cv=cv, return_estimator=return_estimator
    )
    cv_results_sk = cross_validate(model, X=X, y=y, cv=cv, return_estimator=True)

    assert cv_results_da.keys() == cv_results_sk.keys()


@pytest.mark.parametrize("require_threshold", (True, False))
def test_diff_detector_require_thresholds(require_threshold: bool):
    """
    Should fail if requiring thresholds, but not calling cross_validate
    """
    X = pd.DataFrame(np.random.random((100, 5)))
    y = pd.DataFrame(np.random.random((100, 2)))

    model = DiffBasedAnomalyDetector(
        base_estimator=MultiOutputRegressor(LinearRegression()),
        require_thresholds=require_threshold,
    )

    model.fit(X, y)

    if require_threshold:
        # FAIL: Forgot to call .cross_validate to calculate thresholds.
        with pytest.raises(AttributeError):
            model.anomaly(X, y)

        model.cross_validate(X=X, y=y)
        model.anomaly(X, y)
    else:
        # thresholds not required
        model.anomaly(X, y)
