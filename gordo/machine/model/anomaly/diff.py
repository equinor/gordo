# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from typing import Optional, Union
from datetime import timedelta

from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import TimeSeriesSplit, cross_validate

from gordo.machine.model.base import GordoBase
from gordo.machine.model import utils as model_utils
from gordo.machine.model.models import KerasAutoEncoder
from gordo.machine.model.anomaly.base import AnomalyDetectorBase


class DiffBasedAnomalyDetector(AnomalyDetectorBase):
    def __init__(
        self,
        base_estimator: BaseEstimator = KerasAutoEncoder(kind="feedforward_hourglass"),
        scaler: TransformerMixin = RobustScaler(),
        require_thresholds: bool = True,
        window=None,
    ):
        """
        Classifier which wraps a ``base_estimator`` and provides a diff error
        based approach to anomaly detection.

        It trains a ``scaler`` to the target **after** training, purely for
        error calculations. The underlying ``base_estimator`` is trained
        with the original, unscaled, ``y``.

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
            Requires calculating ``thresholds_`` via a call to :func:`~DiffBasedAnomalyDetector.cross_validate`.
            If this is set (default True), but :func:`~DiffBasedAnomalyDetector.cross_validate`
            was not called before calling :func:`~DiffBasedAnomalyDetector.anomaly` an ``AttributeError``
            will be raised.
        window: int
            Window size for smoothed thresholds
        """
        self.base_estimator = base_estimator
        self.scaler = scaler
        self.require_thresholds = require_thresholds
        self.window = window

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
                {"scaler": str(self.scaler), "base_estimator": str(self.base_estimator)}
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
        params = {"base_estimator": self.base_estimator, "scaler": self.scaler}
        if self.window is not None:
            params["window"] = self.window
        return params

    def fit(self, X: np.ndarray, y: np.ndarray):
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
        Run cross validation on the model, and will update the model's threshold value based
        on the cross validation folds

        Parameters
        ----------
        X: Union[pd.DataFrame, np.ndarray]
            Input data to the model
        y: Union[pd.DataFrame, np.ndarray]
            Target data
        kwargs: dict
            Any additional kwargs to be passed to :func:`sklearn.model_selection.cross_validate`

        Returns
        -------
        dict
        """
        # Depend on having the trained fold models
        kwargs.update(dict(return_estimator=True, cv=cv))

        cv_output = cross_validate(self, X=X, y=y, **kwargs)

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
            mae = pd.DataFrame(np.abs(y_pred - y_true))

            # For the aggregate threshold for the fold model, use the mse of scaled residuals per timestep
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
        scaled_y_true = model.scaler.transform(y_true)
        scaled_y_pred = model.scaler.transform(y_pred)
        mse_per_time_step = ((scaled_y_pred - scaled_y_true) ** 2).mean(axis=1)
        return pd.Series(mse_per_time_step)

    def anomaly(
        self, X: pd.DataFrame, y: pd.DataFrame, frequency: Optional[timedelta] = None
    ) -> pd.DataFrame:
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
        # Ensure to offset the y to match model out, which could be less if it was an LSTM
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

        if self.window is not None:
            # Calculate scaled tag-level smoothed anomaly scores
            smooth_tag_anomaly_scaled = tag_anomaly_scaled.rolling(self.window).median()
            smooth_tag_anomaly_scaled.columns = smooth_tag_anomaly_scaled.columns.set_levels(
                ["smooth-tag-anomaly-scaled"], level=0
            )
            data = data.join(smooth_tag_anomaly_scaled)

            # Calculate scaled smoothed total anomaly score
            data["smooth-total-anomaly-scaled"] = (
                data["total-anomaly-scaled"].rolling(self.window).median()
            )

            # Calculate unscaled tag-level smoothed anomaly scores
            smooth_tag_anomaly_unscaled = unscaled_abs_diff.rolling(
                self.window
            ).median()
            smooth_tag_anomaly_unscaled.columns = smooth_tag_anomaly_unscaled.columns.set_levels(
                ["smooth-tag-anomaly-unscaled"], level=0
            )
            data = data.join(smooth_tag_anomaly_unscaled)

            # Calculate unscaled smoothed total anomaly score
            data["smooth-total-anomaly-unscaled"] = (
                data["total-anomaly-unscaled"].rolling(self.window).median()
            )

        # If we have `thresholds_` values, then we can calculate anomaly confidence
        confidence, index = None, None
        if (
            hasattr(self, "smooth_feature_thresholds_")
            and self.smooth_feature_thresholds_ is not None
        ):
            confidence = (
                data["smooth-tag-anomaly-scaled"].to_numpy()
                / self.smooth_feature_thresholds_.to_numpy()
            )
            index = data["smooth-tag-anomaly-scaled"].index
        elif hasattr(self, "feature_thresholds_"):
            confidence = tag_anomaly_scaled.values / self.feature_thresholds_.values
            index = tag_anomaly_scaled.index

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
        if (
            hasattr(self, "smooth_aggregate_threshold_")
            and self.smooth_aggregate_threshold_ is not None
        ):
            total_anomaly_confidence = (
                data["smooth-total-anomaly-scaled"] / self.smooth_aggregate_threshold_
            )
        elif hasattr(self, "aggregate_threshold_"):
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
                f"`require_thresholds={self.require_thresholds}` however `.cross_validate`"
                f"needs to be called in order to calculate these thresholds before calling `.anomaly`"
            )

        return data
