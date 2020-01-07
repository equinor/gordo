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
        scaler: sklearn.base.TransformerMixn
            Defaults to ``sklearn.preprocessing.RobustScaler``
            Used for transforming model output and the original ``y`` to calculate
            the difference/error in model output vs expected.
        require_thresholds: bool
            Requires calculating ``thresholds_`` via a call to :func:`~DiffBasedAnomalyDetector.cross_validate`.
            If this is set (default True), but :func:`~DiffBasedAnomalyDetector.cross_validate`
            was not called before calling :func:`~DiffBasedAnomalyDetector.anomaly` an ``AttributeError``
            will be raised.
        """
        self.base_estimator = base_estimator
        self.scaler = scaler
        self.require_thresholds = require_thresholds

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
        return {"base_estimator": self.base_estimator, "scaler": self.scaler}

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

        feature_thresholds = pd.DataFrame()
        scaled_mse_per_timestep = pd.Series()

        for i, ((test_idxs, _train_idxs), split_model) in enumerate(
            zip(kwargs["cv"].split(X, y), cv_output["estimator"])
        ):
            y_pred = split_model.predict(
                X.iloc[test_idxs] if isinstance(X, pd.DataFrame) else X[test_idxs]
            )

            # Adjust y_true for any possible model offset in its prediction
            test_idxs = test_idxs[-len(y_pred) :]
            y_true = y.iloc[test_idxs] if isinstance(y, pd.DataFrame) else y[test_idxs]

            # Model's timestep scaled mse
            scaled_mse = self._scaled_mse_per_timestep(split_model, y_true, y_pred)
            scaled_mse_per_timestep = pd.concat((scaled_mse_per_timestep, scaled_mse))

            # Accumulate the rolling mins of diffs into common df
            tag_thresholds_fold = self._feature_fold_thresholds(y_true, y_pred, fold=i)
            feature_thresholds = feature_thresholds.append(tag_thresholds_fold)

        # Calculate the final thresholds per feature based on the previous fold calculations
        self.feature_thresholds_ = self._final_thresholds(thresholds=feature_thresholds)

        # For the aggregate, use the accumulated mse of scaled residuals per timestep
        self.aggregate_threshold_ = scaled_mse_per_timestep.rolling(6).min().max()
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

    @staticmethod
    def _feature_fold_thresholds(
        y_true: np.ndarray, y_pred: np.ndarray, fold: int
    ) -> pd.Series:
        """
        Calculate the per fold thresholds

        Parameters
        ----------
        y_true: np.ndarray
            True valudes
        y_pred: np.ndarray
            Predicted values
        fold: int
            Current fold iteration number

        Returns
        -------
        pd.Series
            Per feature calculated thresholds
        """
        diff = pd.DataFrame(np.abs(y_pred - y_true)).rolling(6).min().max()
        diff.name = f"fold-{fold}"
        return diff

    @staticmethod
    def _final_thresholds(thresholds: pd.DataFrame) -> pd.Series:
        """
        Calculate the aggregate and final thresholds from previously
        calculated fold thresholds.

        Parameters
        ----------
        thresholds: pd.DataFrame
            Aggregate thresholds from previous folds.

        Returns
        -------
        pd.Series
            Per feature calculated final thresholds over the fold thresholds
        """
        final_thresholds = thresholds.mean()
        final_thresholds.name = "thresholds"
        return final_thresholds

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

        # Calculate the total anomaly between all tags for the original/untransformed
        # Ensure to offset the y to match model out, which could be less if it was an LSTM
        scaled_y = self.scaler.transform(y)
        scaled_diff = model_out_scaled - scaled_y[-len(data) :, :]
        tag_anomaly_scaled = np.abs(scaled_diff)
        tag_anomaly_scaled.columns = pd.MultiIndex.from_product(
            (("tag-anomaly-scaled",), tag_anomaly_scaled.columns)
        )
        data = data.join(tag_anomaly_scaled)
        data["total-anomaly-scaled"] = np.linalg.norm(
            data["tag-anomaly-scaled"], axis=1
        )

        # Unscaled anomalies: feature-wise and total
        unscaled_abs_diff = pd.DataFrame(
            data=np.abs(data["model-output"].values - y.values[-len(data) :, :]),
            index=data.index,
            columns=pd.MultiIndex.from_product(
                (("tag-anomaly-unscaled",), y.columns.tolist())
            ),
        )
        data = data.join(unscaled_abs_diff)
        data["total-anomaly-unscaled"] = np.linalg.norm(
            data["tag-anomaly-unscaled"], axis=1
        )

        # If we have `thresholds_` values, then we can calculate anomaly confidence
        if hasattr(self, "feature_thresholds_"):
            confidence = tag_anomaly_scaled.values / self.feature_thresholds_.values

            # Dataframe of % abs_diff is of the thresholds
            anomaly_confidence_scores = pd.DataFrame(
                confidence,
                index=tag_anomaly_scaled.index,
                columns=pd.MultiIndex.from_product(
                    (("anomaly-confidence",), data["model-output"].columns)
                ),
            )
            data = data.join(anomaly_confidence_scores)

        if hasattr(self, "aggregate_threshold_"):
            scaled_mse = (scaled_diff ** 2).mean(axis=1)
            data["total-anomaly-confidence"] = scaled_mse / self.aggregate_threshold_

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
