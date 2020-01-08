# -*- coding: utf-8 -*-

from gordo.machine.dataset import datasets


def _get_dataset(config):
    """
    Return a GordoBaseDataSet object of a certain type, given a config dict
    """
    dataset_config = dict(config)
    kind = dataset_config.pop("type", "")
    Dataset = getattr(datasets, kind, None)
    if Dataset is None:
        raise ValueError(f'Dataset type "{kind}" is not supported!')

    return Dataset(**dataset_config)
