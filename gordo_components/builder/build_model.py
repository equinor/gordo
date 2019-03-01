# -*- coding: utf-8 -*-

import ast
import os
import logging
import datetime
import time

from typing import Dict, Any, Union
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score, TimeSeriesSplit

from gordo_components import serializer, __version__
from gordo_components.dataset import _get_dataset
from gordo_components.dataset.base import GordoBaseDataset


logger = logging.getLogger(__name__)


def build_model(
    model_config: dict, data_config: Union[GordoBaseDataset, dict], metadata: dict
):
    """
    Build a model and serialize to a directory for later serving.

    Parameters
    ----------
        model_config: dict - mapping of Model to initialize and any additional
                             kwargs which are to be used in it's initialization
                             Example: {'type': 'KerasAutoEncoder',
                                       'kind': 'feedforward_hourglass'}
        data_config: dict - mapping of the Dataset to initialize, following the
                            same logic as model_config
        metadata: dict - mapping of arbitrary metadata data
    Returns
    -------
        Tuple[sklearn.base.BaseEstimator, dict]
    """
    # Get the dataset from config
    logger.debug(f"Initializing Dataset with config {data_config}")

    dataset = (
        data_config
        if isinstance(data_config, GordoBaseDataset)
        else _get_dataset(data_config)
    )

    logger.debug("Fetching training data")
    start = time.time()
    X, y = dataset.get_data()
    end = time.time()
    time_elapsed_data = end - start

    # Get the model and dataset
    logger.debug(f"Initializing Model with config: {model_config}")
    model = serializer.pipeline_from_definition(model_config)

    # Cross validate
    logger.debug(f"Starting to do cross validation")
    start = time.time()
    cv_scores = cross_val_score(
        model, X, y if y is not None else X, cv=TimeSeriesSplit(n_splits=5)
    )
    cv_time = time.time() - start

    # Train
    logger.debug("Starting to train model.")
    start = time.time()
    model.fit(X, y)
    end = time.time()
    time_elapsed_model = end - start

    metadata = {"user-defined": metadata}
    metadata["dataset"] = dataset.get_metadata()
    utc_dt = datetime.datetime.now(datetime.timezone.utc)
    metadata["model"] = {
        "model-creation-date": str(utc_dt.astimezone()),
        "model-builder-version": __version__,
        "model-config": model_config,
        "data-query-duration-sec": time_elapsed_data,
        "model-training-duration-sec": time_elapsed_model,
        "cross-validation": {
            "time": cv_time,
            "scores": {
                "mean": cv_scores.mean(),
                "std": cv_scores.std(),
                "max": cv_scores.max(),
                "min": cv_scores.min(),
            },
        },
    }
    return model, metadata


def _save_model_for_workflow(model: BaseEstimator, metadata: dict, output_dir: str):
    """
    Save a model according to the expected Argo workflow procedure.

    Parameters
    ----------
    model: BaseEstimator - The model to save to the directory with gordo serializer
    metadata: dict - Various mappings of metadata to save alongside model
    output_dir: str - The directory where to save the model, will create directories if needed

    Returns
    -------
    None
    """
    os.makedirs(output_dir, exist_ok=True)  # Ok if some dirs exist
    serializer.dump(model, output_dir, metadata=metadata)

    # Let argo & subsequent model loader know where the model will be saved.
    with open("/tmp/model-location.txt", "w") as f:
        f.write(output_dir)
