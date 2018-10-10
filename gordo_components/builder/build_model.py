# -*- coding: utf-8 -*-

import os
import logging
import joblib

from gordo_components.model import Model
from gordo_components.dataset import Dataset


logger = logging.getLogger(__name__)


def build_model(output_dir, model_config, data_config):

    # Get the model and dataset
    logger.debug("Initializing Model with config: {}".format(model_config))
    model = Model.from_config(model_config)

    logger.debug("Initializing Dataset with config {}".format(data_config))
    data = Dataset.from_config(data_config)

    # TODO: Fit to actual data
    logger.debug("Fetching training data")
    X, y = data.get_train()

    logger.debug("Starting to train model.")
    model.fit(X, y)

    outpath = os.path.join(output_dir, 'model.pkl')

    with open('/tmp/model-location.txt', 'w') as f:
        f.write(outpath)

    # TODO: Get better model saving location
    joblib.dump(model, outpath)
