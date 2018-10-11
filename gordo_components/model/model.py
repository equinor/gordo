# -*- coding: utf-8 -*-

import json
import logging

from gordo_components.model._models import GordoKerasModel


logger = logging.getLogger(__name__)

AVAILABLE_MODELS = {
    'keras': GordoKerasModel
}


def get_model(config):
    kind = config.get('type', '').lower()
    Model = AVAILABLE_MODELS.get(kind)
    if Model is None:
        return ValueError(
            'Type of model: "{}" either not provided or not supported'.format(kind)
        )
    return Model(**config)
