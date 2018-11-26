# -*- coding: utf-8 -*-

import os
import json
import logging
from gordo_components import serializer
from gordo_components.model.base import GordoBaseModel

logger = logging.getLogger(__name__)

'''
Convert the standard model into the Seldon model for serving.
'''


class SeldonModel:
    """
    The Seldon model
    """
    def __init__(self):

        # Set log level, defaulting to DEBUG
        log_level = logging.getLevelName(os.getenv('LOG_LEVEL', 'DEBUG').upper())
        if not isinstance(log_level, int):
            log_level = logging.DEBUG
        logger.setLevel(log_level)
        logging.getLogger('gordo_components').setLevel(log_level)

        logger.debug('Loading model...')
        model_location = os.getenv('MODEL_LOCATION')
        print('MODEL_LOCATION value: {}'.format(model_location))
        if model_location is None:
            raise ValueError('Environment variable "MODEL_LOCATION" not set!')
        if not os.path.isdir(model_location):
            raise NotADirectoryError(
                f'The supplied directory: "{model_location}" does not exist!')

        logger.info(f'Loading up serialized model from dir: {model_location}')

        self.model = serializer.load(model_location)
        self.metadata = serializer.load_metadata(model_location)
        logger.info(f'Model loaded successfully, ready to serve predictions!')

    def predict(self, X, feature_names=None):
        logger.debug('Feature names: {}'.format(feature_names))
        return self.model.predict(X)

    def tags(self):
        return self.metadata
