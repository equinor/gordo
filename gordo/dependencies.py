import inject

from gordo_dataset.dependencies import config as dataset_config


def config(binder: inject.Binder):  # pragma: no cover
    binder.install(dataset_config)


def configure_once():  # pragma: no cover
    inject.configure_once(config)
