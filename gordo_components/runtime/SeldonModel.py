# -*- coding: utf-8 -*-

import os
import logging
from gordo_components import serializer

logger = logging.getLogger(__name__)

'''
Convert the standard model into the Seldon model for serving.
'''


class SeldonModel:
    """
    The Seldon model
    """
    def __init__(self):
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
        logger.info(f'Model loaded successfully, ready to serve predictions!')

    def predict(self, X, feature_names=None):
        logger.debug('Feature names: {}'.format(feature_names))
        return self.model.predict(X)
