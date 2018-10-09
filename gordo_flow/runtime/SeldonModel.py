# -*- coding: utf-8 -*-

import os
import sys
import logging
from loader import load_model

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
        if model_location is None:
            logger.critical(
                'Environment variable "MODEL_LOCATION" not set, unable to continue!!'
            )
            sys.exit(1)
        self.model = load_model(model_location)

    def predict(self, X, feature_names=None):
        return self.model.predict(X)
