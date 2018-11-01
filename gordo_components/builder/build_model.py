# -*- coding: utf-8 -*-

import os
import logging
import joblib

from gordo_components.model import get_model
from gordo_components.dataset import get_dataset


logger = logging.getLogger(__name__)


def build_model(output_dir, model_config, data_config):

    logger.debug("Initializing Dataset with config {}".format(data_config))
    dataset = get_dataset(data_config)

    # TODO: Fit to actual data
    logger.debug("Fetching training data")
    X, y = dataset.get_data()

    # Get the model and dataset
    logger.debug("Initializing Model with config: {}".format(model_config))
    model_config['n_features'] = X.shape[1]
    model = get_model(model_config)

    logger.debug("Starting to train model.")
    logger.critical(f"X: {X.shape}, y: {y.shape}")
    model.fit(X, y)

    # Bit hacky, need to enforce a way to serialize models in a predictable
    # way. ABC may not enforce the model will _actually_ save correctly.
    filename = 'model.h5' if hasattr(model.model, 'save') else 'model.pkl'
    os.makedirs(output_dir, exist_ok=True)  # Ok if some dirs exist
    outpath = os.path.join(output_dir, filename)

    # TODO: Get better model saving ops.
    # If it's a kera's model, need to serialize that seperately as .h5
    # and then save the rest of the model as .pkl
    if hasattr(model.model, 'save'):
        model.model.save(outpath)
        model.model = None  # Can't pickle Keras models.
        outpath = outpath.replace('.h5', '.pkl')
    joblib.dump(model, outpath)  # Pickle remaining model

    # Let argo know where the model will be saved.
    with open('/tmp/model-location.txt', 'w') as f:
        # Save original suffix, ie. .h5 so loading knows there is an associated
        # .pkl; at least until we have a better serialization approach.
        f.write(os.path.join(output_dir, filename))
