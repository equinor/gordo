# -*- coding: utf-8 -*-

from datetime import timedelta

import pytest
import numpy as np
import pandas as pd
import yaml

from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
from sklearn.model_selection import TimeSeriesSplit, KFold

from gordo import serializer
from gordo.machine.model import utils as model_utils
from gordo.machine.model.base import GordoBase
from gordo.machine.model.anomaly.base import AnomalyDetectorBase
from gordo.machine.model.anomaly.diff import (
    DiffBasedAnomalyDetector,
    DiffBasedKFCVAnomalyDetector,
)


@pytest.mark.parametrize("scaler", (MinMaxScaler(), RobustScaler()))
@pytest.mark.parametrize(
    "index", (range(300), pd.date_range("2019-01-01", "2019-01-30", periods=300))
)
@pytest.mark.parametrize("with_thresholds", (True, False))
@pytest.mark.parametrize("shuffle", (True, False))
def test_diff_detector(scaler, index, with_thresholds: bool, shuffle: bool):
    """
    Test the functionality of the DiffBasedAnomalyDetector
    """

    # Some dataset.
    X, y = (
        pd.DataFrame(np.random.random((300, 3))),
        pd.DataFrame(np.random.random((300, 3))),
    )

    base_estimator = MultiOutputRegressor(estimator=LinearRegression())
    model = DiffBasedAnomalyDetector(
        base_estimator=base_estimator,
        scaler=scaler,
        require_thresholds=with_thresholds,
        shuffle=shuffle,
    )

    assert isinstance(model, AnomalyDetectorBase)

    assert model.get_params() == dict(
        base_estimator=base_estimator, scaler=scaler, shuffle=shuffle
    )

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
        assert anomaly_df["anomaly-confidence"].notnull().to_numpy().any()
        assert anomaly_df["total-anomaly-confidence"].notnull().to_numpy().all()

    else:
        assert "anomaly-confidence" not in anomaly_df.columns
        assert "total-anomaly-confidence" not in anomaly_df.columns


@pytest.mark.parametrize("scaler", (MinMaxScaler(), RobustScaler()))
@pytest.mark.parametrize("len_x_y", (100, 144, 300))
@pytest.mark.parametrize("time_index", (True, False))
@pytest.mark.parametrize("with_thresholds", (True, False))
@pytest.mark.parametrize("shuffle", (True, False))
@pytest.mark.parametrize("window", (None, 144))
@pytest.mark.parametrize("smoothing_method", (None, "smm", "sma", "ewma"))
def test_diff_detector_with_window(
    scaler,
    len_x_y: int,
    time_index: bool,
    with_thresholds: bool,
    shuffle: bool,
    window,
    smoothing_method,
):
    """
    Test the functionality of the DiffBasedAnomalyDetector with window
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
        require_thresholds=with_thresholds,
        shuffle=shuffle,
        window=window,
        smoothing_method=smoothing_method,
    )

    assert isinstance(model, AnomalyDetectorBase)

    if window is None:
        assert model.get_params() == dict(
            base_estimator=base_estimator, scaler=scaler, shuffle=shuffle
        )

    elif window is not None and smoothing_method is None:
        assert model.get_params() == dict(
            base_estimator=base_estimator,
            scaler=scaler,
            shuffle=shuffle,
            window=window,
            smoothing_method="smm",
        )

    else:
        assert model.get_params() == dict(
            base_estimator=base_estimator,
            scaler=scaler,
            shuffle=shuffle,
            window=window,
            smoothing_method=smoothing_method,
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
    if window is not None:
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
    else:
        assert all(
            col in anomaly_df.columns
            for col in (
                "total-anomaly-scaled",
                "total-anomaly-unscaled",
                "tag-anomaly-scaled",
                "tag-anomaly-unscaled",
            )
        )
        assert not any(
            col in base_df.columns
            for col in (
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

    if window is not None:
        if smoothing_method is None or smoothing_method == "smm":
            smooth_feature_error_unscaled = (
                feature_error_unscaled.rolling(model.window).median().dropna()
            )
            smooth_total_anomaly_unscaled = (
                total_anomaly_unscaled.rolling(model.window).median().dropna()
            )

        elif smoothing_method == "sma":
            smooth_feature_error_unscaled = (
                feature_error_unscaled.rolling(model.window).mean().dropna()
            )
            smooth_total_anomaly_unscaled = (
                total_anomaly_unscaled.rolling(model.window).mean().dropna()
            )
        elif smoothing_method == "ewma":
            smooth_feature_error_unscaled = feature_error_unscaled.ewm(
                span=model.window
            ).mean()
            smooth_total_anomaly_unscaled = total_anomaly_unscaled.ewm(
                span=model.window
            ).mean()

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

    if window is not None:
        if smoothing_method is None or smoothing_method == "smm":
            smooth_feature_error_scaled = (
                feature_error_scaled.rolling(model.window).median().dropna()
            )
            smooth_total_anomaly_scaled = (
                total_anomaly_scaled.rolling(model.window).median().dropna()
            )
        elif smoothing_method == "sma":
            smooth_feature_error_scaled = (
                feature_error_scaled.rolling(model.window).mean().dropna()
            )
            smooth_total_anomaly_scaled = (
                total_anomaly_scaled.rolling(model.window).mean().dropna()
            )
        elif smoothing_method == "ewma":
            smooth_feature_error_scaled = feature_error_scaled.ewm(
                span=model.window
            ).mean()
            smooth_total_anomaly_scaled = total_anomaly_scaled.ewm(
                span=model.window
            ).mean()

        assert np.allclose(
            smooth_feature_error_scaled.to_numpy(),
            anomaly_df["smooth-tag-anomaly-scaled"].dropna().to_numpy(),
        )
        assert np.allclose(
            smooth_total_anomaly_scaled.to_numpy(),
            anomaly_df["smooth-total-anomaly-scaled"].dropna().to_numpy(),
        )

    # Check number of NA's is consistent with window size
    if (
        smoothing_method != "ewma"
        and model.window is not None
        and len_x_y >= model.window
    ):
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
        assert anomaly_df["anomaly-confidence"].notnull().to_numpy().all()
        assert anomaly_df["total-anomaly-confidence"].notnull().to_numpy().all()
    else:
        assert "anomaly-confidence" not in anomaly_df.columns
        assert "total-anomaly-confidence" not in anomaly_df.columns


@pytest.mark.parametrize("scaler", (MinMaxScaler(), RobustScaler()))
@pytest.mark.parametrize(
    "index", (range(300), pd.date_range("2019-01-01", "2019-01-30", periods=300))
)
@pytest.mark.parametrize("with_thresholds", (True, False))
@pytest.mark.parametrize("shuffle", (True, False))
@pytest.mark.parametrize("window", (6, 144))
@pytest.mark.parametrize("smoothing_method", ("smm", "sma", "ewma"))
@pytest.mark.parametrize("threshold_percentile", (0.975, 1.0))
def test_diff_kfcv_detector(
    scaler,
    index,
    with_thresholds: bool,
    shuffle: bool,
    window: int,
    smoothing_method: str,
    threshold_percentile: float,
):
    """
    Test the functionality of the DiffBasedKFCVAnomalyDetector
    """

    # Some dataset.
    X, y = (
        pd.DataFrame(np.random.random((300, 3))),
        pd.DataFrame(np.random.random((300, 3))),
    )

    base_estimator = MultiOutputRegressor(estimator=LinearRegression())
    model = DiffBasedKFCVAnomalyDetector(
        base_estimator=base_estimator,
        scaler=scaler,
        require_thresholds=with_thresholds,
        shuffle=shuffle,
        window=window,
        smoothing_method=smoothing_method,
        threshold_percentile=threshold_percentile,
    )

    assert isinstance(model, AnomalyDetectorBase)

    assert model.get_params() == dict(
        base_estimator=base_estimator,
        scaler=scaler,
        window=window,
        smoothing_method=smoothing_method,
        shuffle=shuffle,
        threshold_percentile=threshold_percentile,
    )

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
        metadata = model.get_metadata()
        assert not any(np.isnan(metadata["feature-thresholds"]))
        assert not np.isnan(metadata["aggregate-threshold"])
        assert "anomaly-confidence" in anomaly_df.columns
        assert "total-anomaly-confidence" in anomaly_df.columns
        assert anomaly_df["anomaly-confidence"].notnull().to_numpy().all()
        assert anomaly_df["total-anomaly-confidence"].notnull().to_numpy().all()
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
        """
        gordo.machine.model.anomaly.diff.DiffBasedKFCVAnomalyDetector:
            base_estimator:
                sklearn.compose.TransformedTargetRegressor:
                    transformer:
                        sklearn.preprocessing.MinMaxScaler
                    regressor:
                        sklearn.pipeline.Pipeline:
                            steps:
                            - sklearn.preprocessing.MinMaxScaler
                            - gordo.machine.model.models.KerasAutoEncoder:
                                kind: feedforward_hourglass
                                batch_size: 128
                                compression_factor: 0.5
                                encoding_layers: 1
                                func: tanh
                                out_func: linear
                                optimizer: Adam
                                loss: mse
                                epochs: 1000
                                validation_split: 0.1
                                callbacks:
                                    - tensorflow.keras.callbacks.EarlyStopping:
                                        monitor: val_loss
                                        patience: 10
                                        restore_best_weights: true
            scaler: sklearn.preprocessing.MinMaxScaler
            window: 144
            shuffle: true
            threshold_percentile: 0.975
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
@pytest.mark.parametrize("mode", ("tscv", "kfcv"))
def test_diff_detector_threshold(mode: str, n_features_x: int, n_features_y: int):
    """
    Basic construction logic of thresholds_ attribute in the
    DiffBasedAnomalyDetector and DiffBasedKFCVAnomalyDetector
    """
    X = np.random.random((300, n_features_x))
    y = np.random.random((300, n_features_y))

    base_estimator = MultiOutputRegressor(estimator=LinearRegression())
    if mode == "tscv":
        model = DiffBasedAnomalyDetector(base_estimator=base_estimator)
    elif mode == "kfcv":
        model = DiffBasedKFCVAnomalyDetector(base_estimator=base_estimator)

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
    assert isinstance(model.feature_thresholds_, pd.Series)
    assert len(model.feature_thresholds_) == y.shape[1]
    assert all(model.feature_thresholds_.notna())

    if not isinstance(model, DiffBasedKFCVAnomalyDetector):
        assert hasattr(model, "feature_thresholds_per_fold_")
        assert hasattr(model, "aggregate_thresholds_per_fold_")
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
    if model.window is not None and len_x_y <= model.window:
        assert all(model.smooth_feature_thresholds_.isna())
    else:
        assert all(model.smooth_feature_thresholds_.notna())
    assert isinstance(model.smooth_feature_thresholds_per_fold_, pd.DataFrame)
    assert isinstance(model.smooth_aggregate_thresholds_per_fold_, dict)


@pytest.mark.parametrize("mode", ("tscv", "tscv_win", "kfcv"))
def test_diff_detector_get_metadata(mode):
    """
    Test if metadata is generated as expected.
    """
    X = pd.DataFrame(np.random.random((200, 5)))
    y = pd.DataFrame(np.random.random((200, 2)))

    base_estimator = MultiOutputRegressor(LinearRegression())
    if mode == "tscv":
        model = DiffBasedAnomalyDetector(base_estimator=base_estimator)
    elif mode == "tscv_win":
        model = DiffBasedAnomalyDetector(base_estimator=base_estimator, window=144)
    elif mode == "kfcv":
        model = DiffBasedKFCVAnomalyDetector(base_estimator=base_estimator)

    metadata = model.get_metadata()
    assert isinstance(metadata, dict)

    if not isinstance(model, GordoBase):
        assert "base_estimator" in metadata.keys()
        assert "scaler" in metadata.keys()
        assert "shuffle" in metadata.keys()

    # When initialized it should not have a threshold calculated.
    assert "feature-thresholds" not in metadata.keys()
    assert "aggregate-threshold" not in metadata.keys()
    assert "feature-thresholds-per-fold" not in metadata.keys()
    assert "aggregate-thresholds-per-fold" not in metadata.keys()

    # Calling cross validate should set the threshold for it.
    model.cross_validate(X=X, y=y)
    metadata = model.get_metadata()

    if not isinstance(model, GordoBase):
        assert "base_estimator" in metadata.keys()
        assert "scaler" in metadata.keys()
        assert "shuffle" in metadata.keys()

    # Now we have calculated thresholds based on cross validation folds
    assert "feature-thresholds" in metadata.keys()
    assert "aggregate-threshold" in metadata.keys()
    assert isinstance(metadata["feature-thresholds"], list)
    assert len(metadata["feature-thresholds"]) == y.shape[1]

    if mode != "kfcv":
        assert "feature-thresholds-per-fold" in metadata.keys()
        assert "aggregate-thresholds-per-fold" in metadata.keys()
    if mode != "tscv":
        assert "window" in metadata.keys()
        assert "smoothing-method" in metadata.keys()
    if mode == "tscv_win":
        assert "smooth-feature-thresholds" in metadata.keys()
        assert "smooth-aggregate-threshold" in metadata.keys()
        assert "smooth-feature-thresholds-per-fold" in metadata.keys()
        assert "smooth-aggregate-thresholds-per-fold" in metadata.keys()
    if mode == "kfcv":
        assert "threshold-percentile" in metadata.keys()


@pytest.mark.parametrize("return_estimator", (True, False))
@pytest.mark.parametrize("n_features_y", (1, 10))
@pytest.mark.parametrize("mode", ("tscv", "kfcv"))
def test_diff_detector_cross_validate(
    mode: str, n_features_y: int, return_estimator: bool
):
    """
    DiffBasedAnomalyDetector.cross_validate implementation should be the
    same as sklearn.model_selection.cross_validate if called the same.

    And it always will update `return_estimator` to True, as it requires
    the intermediate models to calculate the thresholds
    """
    X = np.random.random((100, 10))
    y = np.random.random((100, n_features_y))

    base_estimator = MultiOutputRegressor(LinearRegression())
    if mode == "tscv":
        model = DiffBasedAnomalyDetector(base_estimator=base_estimator)
        cv = TimeSeriesSplit(n_splits=3)
    elif mode == "kfcv":
        model = DiffBasedKFCVAnomalyDetector(base_estimator=base_estimator)
        cv = KFold(n_splits=5, shuffle=True, random_state=0)

    cv_results_da = model.cross_validate(
        X=X, y=y, cv=cv, return_estimator=return_estimator
    )
    cv_results_sk = cross_validate(model, X=X, y=y, cv=cv, return_estimator=True)

    assert cv_results_da.keys() == cv_results_sk.keys()


@pytest.mark.parametrize("require_threshold", (True, False))
@pytest.mark.parametrize("mode", ("tscv", "kfcv"))
def test_diff_detector_require_thresholds(mode: str, require_threshold: bool):
    """
    Should fail if requiring thresholds, but not calling cross_validate
    """
    X = pd.DataFrame(np.random.random((200, 5)))
    y = pd.DataFrame(np.random.random((200, 2)))

    base_estimator = MultiOutputRegressor(LinearRegression())
    if mode == "tscv":
        model = DiffBasedAnomalyDetector(
            base_estimator=base_estimator, require_thresholds=require_threshold
        )
    elif mode == "kfcv":
        model = DiffBasedKFCVAnomalyDetector(
            base_estimator=base_estimator, require_thresholds=require_threshold
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
