# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from typing import Optional, Union
from datetime import timedelta

from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import TimeSeriesSplit, cross_validate

from gordo_components.model.base import GordoBase
from gordo_components.model import utils as model_utils
from gordo_components.model.models import KerasAutoEncoder
from gordo_components.model.anomaly.base import AnomalyDetectorBase


class DiffBasedAnomalyDetector(AnomalyDetectorBase):
    def __init__(
        self,
        base_estimator: BaseEstimator = KerasAutoEncoder(kind="feedforward_hourglass"),
        scaler: TransformerMixin = RobustScaler(),
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
            defaults to py:class:`gordo_components.model.models.KerasAutoEncoder` with
            ``kind='feedforward_hourglass``
        scaler: sklearn.base.TransformerMixn
            Defaults to ``sklearn.preprocessing.RobustScaler``
            Used for transforming model output and the original ``y`` to calculate
            the difference/error in model output vs expected.
        """
        self.base_estimator = base_estimator
        self.scaler = scaler

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
        if isinstance(self.base_estimator, GordoBase):
            return self.base_estimator.get_metadata()
        else:
            return {
                "scaler": str(self.scaler),
                "base_estimator": str(self.base_estimator),
            }

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
        y: Optional[Union[pd.DataFrame, np.ndarray]] = None,
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
        y: Optional[Union[pd.DataFrame, np.ndarray]]
            Target data
        kwargs: dict
            Any additional kwargs to be passed to :func:`sklearn.model_selection.cross_validate`

        Returns
        -------
        dict
        """
        # Depend on having the trained fold models
        kwargs.update(dict(return_estimator=True, cv=cv))

        X = X if not isinstance(X, pd.DataFrame) else X.values
        y = y if not isinstance(y, pd.DataFrame) else y.values

        # Base cv_output dict, which we'll supplement
        cv_output = cross_validate(self, X=X, y=y, **kwargs)

        thresholds = pd.DataFrame()

        for i, ((test_idxs, _train_idxs), model) in enumerate(
            zip(kwargs["cv"].split(X, y), cv_output["estimator"])
        ):
            y_pred = model.predict(X[test_idxs])
            y_true = y[test_idxs]

            diff = self._fold_thresholds(y_true=y_true, y_pred=y_pred, fold=i)

            # Accumulate the rolling mins of diffs into common df
            thresholds = thresholds.append(diff)

        # And finally, get the mean of all the pre-calculated thresholds over the folds
        self.thresholds_ = self._final_thresholds(thresholds=thresholds)
        return cv_output

    @staticmethod
    def _fold_thresholds(
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
        diff = (
            pd.DataFrame(np.abs(y_pred - y_true[-len(y_pred) :])).rolling(6).min().max()
        )
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

    def _apply_theshold_anomlay_features(self):
        """

        Returns
        -------

        """

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
        tag_anomaly = np.abs(model_out_scaled - scaled_y[-len(data) :, :])
        tag_anomaly.columns = pd.MultiIndex.from_tuples(
            ("tag-anomaly", col) for col in tag_anomaly.columns
        )

        data = data.join(tag_anomaly)

        # Calculate the total anomaly
        data["total-anomaly"] = np.linalg.norm(data["tag-anomaly"], axis=1)

        # If we have `thresholds_` values, then we can calculate anomaly confidence
        if hasattr(self, "thresholds_"):
            y = y if not hasattr(y, "values") else y.values
            model_output = data["model-output"].values
            abs_diff = np.abs(model_output - y[-len(model_output) :])
            confidence_percentage = np.clip(abs_diff / self.thresholds_.values, 0, 1)

            # Dataframe of % abs_diff is of the thresholds
            anomaly_confidence_scores = pd.DataFrame(
                confidence_percentage,
                columns=pd.MultiIndex.from_product(
                    (("anomaly-confidence",), data["model-output"].columns)
                ),
            )
            data = data.join(anomaly_confidence_scores)

        return data
