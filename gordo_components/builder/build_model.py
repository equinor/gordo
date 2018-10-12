# -*- coding: utf-8 -*-

import os
import logging
import joblib

from gordo_components.model import get_model
from gordo_components.dataset import Dataset


logger = logging.getLogger(__name__)


def build_model(output_dir, model_config, data_config):

    logger.debug("Initializing Dataset with config {}".format(data_config))
    data = Dataset(**data_config)

    # TODO: Fit to actual data
    logger.debug("Fetching training data")
    X, y = data.get_train()

    # Get the model and dataset
    logger.debug("Initializing Model with config: {}".format(model_config))
    model_config['n_features'] = X.shape[1]
    model = get_model(model_config)

    logger.debug("Starting to train model.")
    model.fit(X, y)

    filename = 'model.h5' if hasattr(model._model, 'save') else 'model.pkl'
    outpath = os.path.join(output_dir, filename)

    # TODO: Get better model saving ops.
    # If it's a kera's model, need to serialize that seperately as .h5
    # and then save the rest of the model as .pkl
    if hasattr(model._model, 'save'):
        model._model.save(outpath)
        model._model = None  # Can't pickle Keras models.
        outpath = outpath.replace('.h5', '.pkl')
    joblib.dump(model, outpath)  # Pickle remaining model

    # Let argo know where the model will be saved.
    with open('/tmp/model-location.txt', 'w') as f:
        # Save original suffix, ie. .h5 so loading knows there is an associated
        # .pkl; at least until we have a better serialization approach.
        f.write(os.path.join(output_dir, filename))
