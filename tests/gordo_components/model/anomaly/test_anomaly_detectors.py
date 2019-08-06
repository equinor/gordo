# -*- coding: utf-8 -*-

from datetime import timedelta

import pytest
import numpy as np
import pandas as pd
import yaml

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, RobustScaler

from gordo_components import serializer
from gordo_components.model import utils as model_utils
from gordo_components.model.anomaly import DiffBasedAnomalyDetector


@pytest.mark.parametrize("scaler", (MinMaxScaler(), RobustScaler()))
@pytest.mark.parametrize(
    "index", (range(10), pd.date_range("2019-01-01", "2019-01-30", periods=10))
)
@pytest.mark.parametrize("lookback", (0, 5))
def test_diff_detector(scaler, index, lookback):
    """
    Test the functionality of the DiffBasedAnomalyDetector
    """

    # Some dataset.
    X, y = (
        pd.DataFrame(np.random.random((10, 3))),
        pd.DataFrame(np.random.random((10, 3))),
    )

    # Use PCA because it auto adjusts to model input/output like our gordo models
    base_estimator = PCA()

    # Give PCA a .predict method for the anomaly detector
    base_estimator.predict = base_estimator.transform

    model = DiffBasedAnomalyDetector(base_estimator=base_estimator, scaler=scaler)
    assert model.get_params() == dict(base_estimator=base_estimator, scaler=scaler)

    model.fit(X, y)

    output: np.ndarray = model.predict(X)
    base_df = model_utils.make_base_dataframe(
        tags=["A", "B", "C"], model_input=X, model_output=output, index=index
    )

    # Base prediction dataframe has none of these columns
    assert not any(col in base_df.columns for col in ("total-anomaly", "tag-anomaly"))

    # Apply the anomaly detection logic on the base prediction df
    anomaly_df = model.anomaly(X, y, timedelta(days=1))

    # Should have these added error calculated columns now.
    assert all(col in anomaly_df.columns for col in ("total-anomaly", "tag-anomaly"))

    # Verify calculations
    feature_wise_error = np.abs(
        scaler.transform(base_df["model-output"].values) - scaler.transform(y)
    )
    assert np.allclose(feature_wise_error, anomaly_df["tag-anomaly"].values)

    total_anomaly = np.linalg.norm(feature_wise_error, axis=1)
    assert np.allclose(total_anomaly, anomaly_df["total-anomaly"].values)


@pytest.mark.parametrize(
    "config",
    (
        """
        gordo_components.model.anomaly.diff.DiffBasedAnomalyDetector:
            base_estimator:
                sklearn.pipeline.Pipeline:
                    steps:
                        - sklearn.preprocessing.data.MinMaxScaler
                        - gordo_components.model.models.KerasAutoEncoder:
                            kind: feedforward_hourglass
                    memory:
    """,
        """
        gordo_components.model.anomaly.diff.DiffBasedAnomalyDetector:
            base_estimator:
                gordo_components.model.models.KerasAutoEncoder:
                    kind: feedforward_hourglass
    """,
    ),
)
def test_diff_detector_serializability(config):
    """
    Should play well with the gordo serializer
    """
    config = yaml.load(config)

    model = serializer.pipeline_from_definition(config)
    serializer.pipeline_into_definition(model)
    serialized_bytes = serializer.dumps(model)
    serializer.loads(serialized_bytes)
