# -*- coding: utf-8 -*-

import logging

from gordo_components.model import models


logger = logging.getLogger(__name__)


def get_model(config):
    type = config.get("type", "")
    Model = getattr(models, type, None)
    if Model is None:
        raise ValueError(
            f'Type of model: "{type}" either not provided or not supported'
        )
    return Model(**config)
