# -*- coding: utf-8 -*-

import os
from .loader import load_model

'''
Convert the standard model into the Seldon model for serving.
'''


class SeldonModel:
    """
    The Seldon model
    """
    def __init__(self):
        self.model = load_model(os.getenv('MODEL_LOCATION'))

    def predict(self, X, feature_names=None):
        return self.model.predict(X)
