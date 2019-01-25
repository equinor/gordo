# -*- coding: utf-8 -*-

from gordo_components.dataset import _datasets


def get_dataset(config):
    """
    Return a GordoBaseDataSet object of a certain type, given a config dict
    """
    kind = config.get("type", "")
    Dataset = getattr(_datasets, kind, None)
    if Dataset is None:
        raise ValueError('Dataset type "{}" is not supported!'.format(kind))
    return Dataset(**config)
