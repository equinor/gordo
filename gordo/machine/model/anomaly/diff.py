# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import xarray as xr

from typing import Optional, Union
from datetime import timedelta

from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import TimeSeriesSplit, KFold, cross_validate as c_val
from sklearn.utils import shuffle
from sklearn.exceptions import NotFittedError

from gordo.machine.model.base import GordoBase
from gordo.machine.model import utils as model_utils
from gordo.machine.model.models import KerasAutoEncoder
from gordo.machine.model.anomaly.base import AnomalyDetectorBase


class DiffBasedAnomalyDetector(AnomalyDetectorBase):
    def __init__(
        self,
        base_estimator: BaseEstimator = KerasAutoEncoder(kind="feedforward_hourglass"),
        scaler: TransformerMixin = MinMaxScaler(),
        require_thresholds: bool = True,
        shuffle: bool = False,
        window: Optional[int] = None,
        smoothing_method: Optional[str] = None,
    ):
        """
        Estimator which wraps a ``base_estimator`` and provides a diff error
        based approach to anomaly detection.

        It trains a ``scaler`` to the target **after** training, purely for
        error calculations. The underlying ``base_estimator`` is trained
        with the original, unscaled, ``y``.

        Threshold calculation is based on a rolling statistic of the validation errors
        on the last fold of cross-validation.

        Parameters
        ----------
        base_estimator: sklearn.base.BaseEstimator
            The model to which normal ``.fit``, ``.predict`` methods will be used.
            defaults to py:class:`gordo.machine.model.models.KerasAutoEncoder` with
            ``kind='feedforward_hourglass``
        scaler: sklearn.base.TransformerMixin
            Defaults to ``sklearn.preprocessing.RobustScaler``
            Used for transforming model output and the original ``y`` to calculate
            the difference/error in model output vs expected.
        require_thresholds: bool
            Requires calculating ``thresholds_`` via a call to
            :func:`~DiffBasedAnomalyDetector.cross_validate`. If this is set
            (default True), but :func:`~DiffBasedAnomalyDetector.cross_validate` was not
            called before calling :func:`~DiffBasedAnomalyDetector.anomaly`
            an ``AttributeError`` will be raised.
        shuffle: bool
            Flag to shuffle or not data in ``.fit`` so that the model, if relevant,
            will be trained on a sample of data accross the time range and not just
            the last elements according to model arg ``validation_split``.
        window: int
            Window size for smoothed thresholds
        smoothing_method: str
            Method to be used together with ``window`` to smooth metrics.
            Must be one of: 'smm': simple moving median, 'sma': simple moving average or
            'ewma': exponential weighted moving average.
        """
        self.base_estimator = base_estimator
        self.scaler = scaler
        self.require_thresholds = require_thresholds
        self.shuffle = shuffle
        self.window = window
        self.smoothing_method = smoothing_method
        if self.window is not None and self.smoothing_method is None:
            self.smoothing_method = "smm"

    def __getattr__(self, item):
        """
        Treat this model as transparent into base_estimator unless
        referring to something owned by this object
        """
        if item in self.__dict__:
            return getattr(self, item)
        else:
            return getattr(self.base_estimator, item)

    def get_metadata(self):
        """
        Generates model metadata.

        Returns
        -------
        dict
        """
        metadata = dict()
        if hasattr(self, "feature_thresholds_"):
            metadata["feature-thresholds"] = self.feature_thresholds_.tolist()
        if hasattr(self, "aggregate_threshold_"):
            metadata["aggregate-threshold"] = self.aggregate_threshold_
        if hasattr(self, "feature_thresholds_per_fold_"):
            metadata[
                "feature-thresholds-per-fold"
            ] = self.feature_thresholds_per_fold_.to_dict()
        if hasattr(self, "aggregate_thresholds_per_fold_"):
            metadata[
                "aggregate-thresholds-per-fold"
            ] = self.aggregate_thresholds_per_fold_
        # Window threshold metadata
        if hasattr(self, "window"):
            metadata["window"] = self.window
        if hasattr(self, "smoothing_method"):
            metadata["smoothing-method"] = self.smoothing_method
        if (
            hasattr(self, "smooth_feature_thresholds_")
            and self.smooth_aggregate_threshold_ is not None
        ):
            metadata[
                "smooth-feature-thresholds"
            ] = self.smooth_feature_thresholds_.tolist()
        if (
            hasattr(self, "smooth_aggregate_threshold_")
            and self.smooth_aggregate_threshold_ is not None
        ):
            metadata["smooth-aggregate-threshold"] = self.smooth_aggregate_threshold_

        if hasattr(self, "smooth_feature_thresholds_per_fold_"):
            metadata[
                "smooth-feature-thresholds-per-fold"
            ] = self.smooth_feature_thresholds_per_fold_.to_dict()
        if hasattr(self, "smooth_aggregate_thresholds_per_fold_"):
            metadata[
                "smooth-aggregate-thresholds-per-fold"
            ] = self.smooth_aggregate_thresholds_per_fold_

        if isinstance(self.base_estimator, GordoBase):
            metadata.update(self.base_estimator.get_metadata())
        else:
            metadata.update(
                {
                    "scaler": str(self.scaler),
                    "base_estimator": str(self.base_estimator),
                    "shuffle": self.shuffle,
                }
            )
        return metadata

    def score(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.DataFrame],
        sample_weight: Optional[np.ndarray] = None,
    ) -> float:
        return self.base_estimator.score(X, y)

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Returns
        -------
        dict
        """
        params = {
            "base_estimator": self.base_estimator,
            "scaler": self.scaler,
            "shuffle": self.shuffle,
        }
        if self.window is not None:
            params["window"] = self.window
            params["smoothing_method"] = self.smoothing_method
        return params

    def fit(self, X: np.ndarray, y: np.ndarray):
        if self.shuffle:
            X_shuff, y_shuff = shuffle(X, y, random_state=0)
            self.base_estimator.fit(X_shuff, y_shuff)
        else:
            self.base_estimator.fit(X, y)

        self.scaler.fit(y)  # Scaler is used for calculating errors in .anomaly()
        return self

    def cross_validate(
        self,
        *,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, np.ndarray],
        cv=TimeSeriesSplit(n_splits=3),
        **kwargs,
    ):
        """
        Run TimeSeries cross validation on the model, and will update the model's
        threshold values based on the cross validation folds.

        Parameters
        ----------
        X: Union[pd.DataFrame, np.ndarray]
            Input data to the model
        y: Union[pd.DataFrame, np.ndarray]
            Target data
        kwargs: dict
            Any additional kwargs to be passed to
            :func:`sklearn.model_selection.cross_validate`

        Returns
        -------
        dict
        """
        # Depend on having the trained fold models
        kwargs.update(dict(return_estimator=True, cv=cv))

        cv_output = c_val(self, X=X, y=y, **kwargs)

        self.feature_thresholds_per_fold_ = pd.DataFrame()
        self.aggregate_thresholds_per_fold_ = {}
        self.smooth_feature_thresholds_per_fold_ = pd.DataFrame()
        self.smooth_aggregate_thresholds_per_fold_ = {}
        smooth_aggregate_threshold_fold = None
        smooth_tag_thresholds_fold = None

        for i, ((_, test_idxs), split_model) in enumerate(
            zip(kwargs["cv"].split(X, y), cv_output["estimator"])
        ):
            y_pred = split_model.predict(
                X.iloc[test_idxs] if isinstance(X, pd.DataFrame) else X[test_idxs]
            )

            # Adjust y_true for any possible model offset in its prediction
            test_idxs = test_idxs[-len(y_pred) :]
            y_true = y.iloc[test_idxs] if isinstance(y, pd.DataFrame) else y[test_idxs]

            # Model's timestep scaled mse over all features
            scaled_mse = self._scaled_mse_per_timestep(split_model, y_true, y_pred)

            # Absolute error
            mae = self._absolute_error(y_true, y_pred)

            # For the aggregate threshold for the fold model,
            # use the mse of scaled residuals per timestep
            aggregate_threshold_fold = scaled_mse.rolling(6).min().max()
            self.aggregate_thresholds_per_fold_[f"fold-{i}"] = aggregate_threshold_fold

            # Accumulate the rolling mins of diffs into common df
            tag_thresholds_fold = mae.rolling(6).min().max()
            tag_thresholds_fold.name = f"fold-{i}"
            self.feature_thresholds_per_fold_ = self.feature_thresholds_per_fold_.append(
                tag_thresholds_fold
            )

            if self.window is not None:
                # Calculate smoothed thresholds only if len of data >= window
                smooth_aggregate_threshold_fold = (
                    scaled_mse.rolling(self.window).min().max()
                )
                self.smooth_aggregate_thresholds_per_fold_[
                    f"fold-{i}"
                ] = smooth_aggregate_threshold_fold

                smooth_tag_thresholds_fold = mae.rolling(self.window).min().max()
                smooth_tag_thresholds_fold.name = f"fold-{i}"
                self.smooth_feature_thresholds_per_fold_ = self.smooth_feature_thresholds_per_fold_.append(
                    smooth_tag_thresholds_fold
                )

        # Final thresholds are the thresholds from the last cv split/fold
        self.feature_thresholds_ = tag_thresholds_fold

        # For the aggregate also use the thresholds from the last split/fold
        self.aggregate_threshold_ = aggregate_threshold_fold

        # For the smoothed thresholds also use the last fold
        self.smooth_aggregate_threshold_ = smooth_aggregate_threshold_fold
        self.smooth_feature_thresholds_ = smooth_tag_thresholds_fold

        return cv_output

    @staticmethod
    def _scaled_mse_per_timestep(
        model: BaseEstimator,
        y_true: Union[pd.DataFrame, np.ndarray],
        y_pred: Union[pd.DataFrame, np.ndarray],
    ) -> pd.Series:
        """
        Calculate the scaled MSE per timestep/sample
        Parameters
        ----------
        model: BaseEstimator
            Instance of a fitted :class:`~DiffBasedAnomalyDetector`
        y_true: Union[numpy.ndarray, pd.DataFrame]
        y_pred: Union[numpy.ndarray, pd.DataFrame]
        Returns
        -------
        panadas.Series
            The MSE calculated from the scaled y and y predicted.
        """
        try:
            scaled_y_true = model.scaler.transform(y_true)
        except (NotFittedError, ValueError):
            scaled_y_true = model.scaler.fit_transform(y_true)
        scaled_y_pred = model.scaler.transform(y_pred)
        mse_per_time_step = ((scaled_y_pred - scaled_y_true) ** 2).mean(axis=1)
        return pd.Series(mse_per_time_step)

    @staticmethod
    def _absolute_error(
        y_true: Union[pd.DataFrame, np.ndarray], y_pred: Union[pd.DataFrame, np.ndarray]
    ) -> pd.DataFrame:

        return pd.DataFrame(np.abs(y_true - y_pred))

    def _smoothing(self, metric: Union[pd.DataFrame, pd.Series]):
        if self.smoothing_method == "smm":
            return metric.rolling(self.window).median()
        elif self.smoothing_method == "sma":
            return metric.rolling(self.window).mean()
        elif self.smoothing_method == "ewma":
            return metric.ewm(span=self.window).mean()

    def anomaly(
        self,
        X: Union[pd.DataFrame, xr.DataArray],
        y: Union[pd.DataFrame, xr.DataArray],
        frequency: Optional[timedelta] = None,
    ) -> Union[pd.DataFrame, xr.Dataset]:
        """
        Create an anomaly dataframe from the base provided dataframe.

        Parameters
        ----------
        X: pd.DataFrame
            Dataframe representing the data to go into the model.
        y: pd.DataFrame
            Dataframe representing the target output of the model.

        Returns
        -------
        pd.DataFrame
            A superset of the original base dataframe with added anomaly specific
            features
        """

        # Get the model output, falling back to transform if 'predict' doesn't exist
        model_output = (
            self.predict(X) if hasattr(self, "predict") else self.transform(X)
        )

        # Create the basic dataframe with 'model-output' & 'model-input'
        data = model_utils.make_base_dataframe(
            tags=X.columns,
            model_input=getattr(X, "values", X),
            model_output=model_output,
            target_tag_list=y.columns,
            index=getattr(X, "index", None),
            frequency=frequency,
        )

        model_out_scaled = pd.DataFrame(
            self.scaler.transform(data["model-output"]),
            columns=data["model-output"].columns,
            index=data.index,
        )

        # Calculate the absolute scaled tag anomaly
        # Ensure to offset the y to match model out, which could be less if it is a LSTM
        scaled_y = self.scaler.transform(y)
        tag_anomaly_scaled = np.abs(model_out_scaled - scaled_y[-len(data) :, :])
        tag_anomaly_scaled.columns = pd.MultiIndex.from_product(
            (("tag-anomaly-scaled",), tag_anomaly_scaled.columns)
        )
        data = data.join(tag_anomaly_scaled)

        # Calculate scaled total anomaly
        data["total-anomaly-scaled"] = np.square(data["tag-anomaly-scaled"]).mean(
            axis=1
        )

        # Calculate the absolute unscaled tag anomalies
        unscaled_abs_diff = pd.DataFrame(
            data=np.abs(
                data["model-output"].to_numpy() - y.to_numpy()[-len(data) :, :]
            ),
            index=data.index,
            columns=pd.MultiIndex.from_product(
                (("tag-anomaly-unscaled",), y.columns.tolist())
            ),
        )
        data = data.join(unscaled_abs_diff)

        # Calculate the scaled total anomaly
        data["total-anomaly-unscaled"] = np.square(data["tag-anomaly-unscaled"]).mean(
            axis=1
        )

        if self.window is not None and self.smoothing_method is not None:
            # Calculate scaled tag-level smoothed anomaly scores
            smooth_tag_anomaly_scaled = self._smoothing(tag_anomaly_scaled)
            smooth_tag_anomaly_scaled.columns = smooth_tag_anomaly_scaled.columns.set_levels(
                ["smooth-tag-anomaly-scaled"], level=0
            )
            data = data.join(smooth_tag_anomaly_scaled)

            # Calculate scaled smoothed total anomaly score
            data["smooth-total-anomaly-scaled"] = self._smoothing(
                data["total-anomaly-scaled"]
            )

            # Calculate unscaled tag-level smoothed anomaly scores
            smooth_tag_anomaly_unscaled = self._smoothing(unscaled_abs_diff)

            smooth_tag_anomaly_unscaled.columns = smooth_tag_anomaly_unscaled.columns.set_levels(
                ["smooth-tag-anomaly-unscaled"], level=0
            )
            data = data.join(smooth_tag_anomaly_unscaled)

            # Calculate unscaled smoothed total anomaly score
            data["smooth-total-anomaly-unscaled"] = self._smoothing(
                data["total-anomaly-unscaled"]
            )

        # If we have `thresholds_` values, then we can calculate anomaly confidence
        confidence, index = None, None

        if hasattr(self, "feature_thresholds_"):
            confidence = unscaled_abs_diff.values / self.feature_thresholds_.values
            index = unscaled_abs_diff.index

        if confidence is not None and index is not None:
            # Dataframe of % abs_diff is of the thresholds
            # This is now based on the smoothed tag anomaly
            anomaly_confidence_scores = pd.DataFrame(
                confidence,
                index=index,
                columns=pd.MultiIndex.from_product(
                    (("anomaly-confidence",), data["model-output"].columns)
                ),
            )
            data = data.join(anomaly_confidence_scores)

        total_anomaly_confidence = None

        if hasattr(self, "aggregate_threshold_"):
            total_anomaly_confidence = (
                data["total-anomaly-scaled"] / self.aggregate_threshold_
            )

        if total_anomaly_confidence is not None:
            data["total-anomaly-confidence"] = total_anomaly_confidence

        # Explicitly raise error if we were required to do threshold based calculations
        # should would have required a call to .cross_validate before .anomaly
        if self.require_thresholds and not any(
            hasattr(self, attr)
            for attr in ("feature_thresholds_", "aggregate_threshold_")
        ):
            raise AttributeError(
                f"`require_thresholds={self.require_thresholds}` however "
                f"`.cross_validate` needs to be called in order to calculate these"
                f"thresholds before calling `.anomaly`"
            )

        return data


class DiffBasedKFCVAnomalyDetector(DiffBasedAnomalyDetector):
    def __init__(
        self,
        base_estimator: BaseEstimator = KerasAutoEncoder(kind="feedforward_hourglass"),
        scaler: TransformerMixin = MinMaxScaler(),
        require_thresholds: bool = True,
        shuffle: bool = True,
        window: int = 144,
        smoothing_method: str = "smm",
        threshold_percentile: float = 0.99,
    ):
        """
        Estimator which wraps a ``base_estimator`` and provides a diff error
        based approach to anomaly detection.

        It trains a ``scaler`` to the target **after** training, purely for
        error calculations. The underlying ``base_estimator`` is trained
        with the original, unscaled, ``y``.

        Threshold calculation is based on a percentile of the smoothed validation
        errors as calculated from cross-validation predictions.

        Parameters
        ----------
        base_estimator: sklearn.base.BaseEstimator
            The model to which normal ``.fit``, ``.predict`` methods will be used.
            defaults to py:class:`gordo.machine.model.models.KerasAutoEncoder` with
            ``kind='feedforward_hourglass``
        scaler: sklearn.base.TransformerMixin
            Defaults to ``sklearn.preprocessing.RobustScaler``
            Used for transforming model output and the original ``y`` to calculate
            the difference/error in model output vs expected.
        require_thresholds: bool
            Requires calculating ``thresholds_`` via a call to
            :func:`~DiffBasedAnomalyDetector.cross_validate`.
            If this is set (default True), but
            :func:`~DiffBasedAnomalyDetector.cross_validate` was not called before
            calling :func:`~DiffBasedAnomalyDetector.anomaly` an ``AttributeError``
            will be raised.
        shuffle: bool
            Flag to shuffle or not data in ``.fit`` so that the model, if relevant,
            will be trained on a sample of data accross the time range and not just
            the last elements according to model arg ``validation_split``.
        window: int
            Window size for smooth metrics and threshold calculation.
        smoothing_method: str
            Method to be used together with ``window`` to smooth metrics.
            Must be one of: 'smm': simple moving median, 'sma': simple moving average or
            'ewma': exponential weighted moving average.
        threshold_percentile: float
            Percentile of the validation data to be used to calculate the threshold.
        """
        self.base_estimator = base_estimator
        self.scaler = scaler
        self.require_thresholds = require_thresholds
        self.window = window
        self.shuffle = shuffle
        self.smoothing_method = smoothing_method
        self.threshold_percentile = threshold_percentile

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Returns
        -------
        dict
        """
        params = {
            "base_estimator": self.base_estimator,
            "scaler": self.scaler,
            "window": self.window,
            "smoothing_method": self.smoothing_method,
            "shuffle": self.shuffle,
            "threshold_percentile": self.threshold_percentile,
        }
        return params

    def get_metadata(self):
        """
        Generates model metadata.

        Returns
        -------
        dict
        """
        metadata = dict()

        if hasattr(self, "feature_thresholds_"):
            metadata["feature-thresholds"] = self.feature_thresholds_.tolist()
        if hasattr(self, "aggregate_threshold_"):
            metadata["aggregate-threshold"] = self.aggregate_threshold_
        if isinstance(self.base_estimator, GordoBase):
            metadata.update(self.base_estimator.get_metadata())
        else:
            metadata.update(
                {
                    "scaler": str(self.scaler),
                    "base_estimator": str(self.base_estimator),
                    "shuffle": self.shuffle,
                    "window": self.window,
                    "smoothing-method": self.smoothing_method,
                    "threshold-percentile": self.threshold_percentile,
                }
            )
        return metadata

    def cross_validate(
        self,
        *,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, np.ndarray],
        cv=KFold(n_splits=5, shuffle=True, random_state=0),
        **kwargs,
    ):
        """
        Run Kfold cross validation on the model, and will update the model's threshold
        values based on a percentile of the validation metrics.

        Parameters
        ----------
        X: Union[pd.DataFrame, np.ndarray]
            Input data to the model
        y: Union[pd.DataFrame, np.ndarray]
            Target data
        kwargs: dict
            Any additional kwargs to be passed to
            :func:`sklearn.model_selection.cross_validate`

        Returns
        -------
        dict
        """

        # Depend on having the trained fold models
        kwargs.update(dict(return_estimator=True, cv=cv))

        cv_output = c_val(self, X=X, y=y, **kwargs)

        # Create empty dataframes to hold fold data
        y_pred = pd.DataFrame(
            np.zeros_like(y),
            index=getattr(y, "index", None),
            columns=getattr(y, "columns", None),
        )
        y = pd.DataFrame(y)
        y_val_mse = pd.Series(index=getattr(y, "index", None))

        # Calculate per-fold validation metrics
        for i, ((_, test_idxs), split_model) in enumerate(
            zip(kwargs["cv"].split(X, y), cv_output["estimator"])
        ):
            y_pred.iloc[test_idxs] = split_model.predict(
                X.iloc[test_idxs].to_numpy()
                if isinstance(X, pd.DataFrame)
                else X[test_idxs]
            )

            y_val_mse.iloc[test_idxs] = self._scaled_mse_per_timestep(
                split_model, y.iloc[test_idxs], y_pred.iloc[test_idxs]
            ).to_numpy()

        # Calculate aggregate threshold
        self.aggregate_threshold_ = self._calculate_threshold(y_val_mse)

        # Calculate tag thresholds
        self.feature_thresholds_ = self._calculate_feature_thresholds(y, y_pred)

        return cv_output

    def _calculate_feature_thresholds(
        self, y_true: pd.DataFrame, y_pred: pd.DataFrame
    ) -> np.ndarray:
        absolute_error = self._absolute_error(y_true, y_pred)
        return self._calculate_threshold(absolute_error)

    def _calculate_threshold(
        self, validation_metric: Union[pd.DataFrame, pd.Series]
    ) -> Union[float, pd.Series]:
        val_metric = self._smoothing(validation_metric)
        return val_metric.quantile(self.threshold_percentile)
