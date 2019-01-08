# -*- coding: utf-8 -*-

import os
import logging
import joblib
import copy
import datetime
import time

from gordo_components import serializer, __version__
from gordo_components.dataset import get_dataset


logger = logging.getLogger(__name__)


def build_model(output_dir: str, model_config: dict, data_config: dict):
    """
    Build a model and serialize to a directory for later serving.

    Parameters
    ----------
        output_dir: str - The path to the directory to save the trained model
        model_config: dict - mapping of Model to initialize and any additional
                             kwargs which are to be used in it's initialization
                             Example: {'type': 'KerasAutoEncoder',
                                       'kind': 'feedforward_symetric'}
        data_config: dict - mapping of the Dataset to initialize, following the
                            same logic as model_config
    Returns
    -------
        None
    """
    # Get the dataset from config
    logger.debug("Initializing Dataset with config {}".format(data_config))
    dataset = get_dataset(data_config)

    logger.debug("Fetching training data")
    start = time.time()
    X, y = dataset.get_data()
    end = time.time()
    time_elapsed_data = end - start

    # Get the model and dataset
    logger.debug("Initializing Model with config: {}".format(model_config))
    model = serializer.pipeline_from_definition(model_config)

    logger.debug("Starting to train model.")
    start = time.time()
    model.fit(X, y)
    end = time.time()
    time_elapsed_model = end - start

    # Save the model/pipeline + metadata
    os.makedirs(output_dir, exist_ok=True)  # Ok if some dirs exist
    logger.debug(f'Saving model to output dir: {output_dir}')
    metadata = {}
    metadata['dataset'] = dataset.get_metadata()
    utc_dt = datetime.datetime.now(datetime.timezone.utc)
    metadata['model'] = {'model_creation_date' : str(utc_dt.astimezone()),
                         'model_builder_version' : __version__,
                         'model_config' : model_config,
                         'data_query_duration_sec' : time_elapsed_data,
                         'model_training_duration_sec' : time_elapsed_model}

    serializer.dump(model, output_dir,metadata=metadata)

    # Let argo & subsequent model loader know where the model will be saved.
    with open('/tmp/model-location.txt', 'w') as f:
        f.write(output_dir)
    logger.info(f'Successfully trained model, dumped to "{output_dir}, exiting.')
