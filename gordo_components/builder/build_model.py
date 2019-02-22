# -*- coding: utf-8 -*-

import ast
import os
import logging
import datetime
import time

from typing import Dict, Any
from sklearn.base import BaseEstimator

from gordo_components import serializer, __version__
from gordo_components.dataset import _get_dataset


logger = logging.getLogger(__name__)


def build_model(model_config: dict, data_config: dict, metadata: dict):
    """
    Build a model and serialize to a directory for later serving.

    Parameters
    ----------
        model_config: dict - mapping of Model to initialize and any additional
                             kwargs which are to be used in it's initialization
                             Example: {'type': 'KerasAutoEncoder',
                                       'kind': 'feedforward_model'}
        data_config: dict - mapping of the Dataset to initialize, following the
                            same logic as model_config
        metadata: dict - mapping of arbitrary metadata data
    Returns
    -------
        Tuple[sklearn.base.BaseEstimator, dict]
    """
    # Get the dataset from config
    logger.debug(f"Initializing Dataset with config {data_config}")
    dataset = _get_dataset(data_config)

    logger.debug("Fetching training data")
    start = time.time()
    X, y = dataset.get_data()
    end = time.time()
    time_elapsed_data = end - start

    # Get the model and dataset
    logger.debug(f"Initializing Model with config: {model_config}")
    model = serializer.pipeline_from_definition(model_config)

    logger.debug("Starting to train model.")
    start = time.time()
    model.fit(X, y)
    end = time.time()
    time_elapsed_model = end - start

    metadata = {"user-defined": metadata}
    metadata["dataset"] = dataset.get_metadata()
    utc_dt = datetime.datetime.now(datetime.timezone.utc)
    metadata["model"] = {
        "model_creation_date": str(utc_dt.astimezone()),
        "model_builder_version": __version__,
        "model_config": model_config,
        "data_query_duration_sec": time_elapsed_data,
        "model_training_duration_sec": time_elapsed_model,
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
