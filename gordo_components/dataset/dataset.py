# -*- coding: utf-8 -*-

from gordo_components.dataset._datasets import (
    RandomDataset, InfluxBackedDataset
)


# TODO: We should get from the module `datasets` directly, as we do with models
AVAILABLE_DATASETS = {
    'random': RandomDataset,
    'influx': InfluxBackedDataset
}


def get_dataset(config):
    """
    Return a GordoBaseDataSet object of a certain type, given a config dict
    """
    kind = config.get('type', '').lower()
    Dataset = AVAILABLE_DATASETS.get(kind)
    if Dataset is None:
        raise ValueError('Dataset type "{}" is not supported!'.format(kind))
    return Dataset(**config)
