# -*- coding: utf-8 -*-

import os
import json

import numpy as np
import pandas as pd

from typing import Optional, Union
from datetime import timedelta

from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin

from gordo_components import serializer
from gordo_components.model.base import GordoBase
from gordo_components.model import utils as model_utils
from gordo_components.model.models import KerasAutoEncoder


class DiffBasedAnomalyDetector(BaseEstimator, GordoBase):
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

    def save_to_dir(self, directory: str):
        serializer.dump(self.base_estimator, os.path.join(directory, "base_estimator"))
        serializer.dump(self.scaler, os.path.join(directory, "scaler"))
        with open(os.path.join(directory, "params.json"), "w") as f:
            params = {
                k: v
                for k, v in self.get_params().items()
                if k not in ("base_estimator", "scaler")
            }
            json.dump(params, f)

    @classmethod
    def load_from_dir(cls, directory: str):
        with open(os.path.join(directory, "params.json"), "r") as f:
            params = json.load(f)
        params["base_estimator"] = serializer.load(
            os.path.join(directory, "base_estimator")
        )
        params["scaler"] = serializer.load(os.path.join(directory, "scaler"))
        return cls(**params)

    def get_params(self, deep=True):
        return {"base_estimator": self.base_estimator, "scaler": self.scaler}

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.base_estimator.fit(X, y)
        self.scaler.fit(y)  # Scaler is used for calculating errors in .anomaly()
        return self

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

        return data
