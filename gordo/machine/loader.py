import yaml

from copy import copy
from typing import TypedDict, Union, cast

from gordo_core.exceptions import ConfigException

from gordo.machine.metadata import Metadata
from gordo.utils import join_json_paths
from gordo_core.base import GordoBaseDataset

from .constants import MACHINE_YAML_FIELDS


class MachineConfigException(ConfigException):
    pass


class _BaseConfig(TypedDict, total=False):
    model: dict
    dataset: Union[dict, GordoBaseDataset]
    evaluation: dict
    metadata: Union[dict, Metadata]
    runtime: dict


class GlobalsConfig(_BaseConfig):
    pass


class MachineConfig(_BaseConfig):
    name: str


class ModelConfig(MachineConfig):
    project_name: str


def _load_config(config: dict, json_path: str = None) -> dict:
    new_config = copy(config)
    for field in config.keys():
        if field in MACHINE_YAML_FIELDS:
            val = config.get(field)
            if type(val) is str:
                try:
                    new_config[field] = yaml.safe_load(val)
                except yaml.YAMLError as e:
                    message = "Error loading YAML from '%s'" % join_json_paths(
                        field, json_path
                    )
                    raise MachineConfigException(message + ": " + str(e))
    return new_config


def load_globals_config(config: dict, json_path: str = None) -> GlobalsConfig:
    """
    Load `GlobalsConfig` from the dict

    Parameters
    ----------
    config: str
        Config to load.
    json_path: str
        JSON path position of the config.

    Returns
    -------

    """
    return cast(GlobalsConfig, _load_config(config, json_path))


def load_machine_config(config: dict, json_path: str = None) -> MachineConfig:
    """
    Load `MachineConfig` from the dict

    Parameters
    ----------
    config: str
        Config to load.
    json_path: str
        JSON path position of the config.

    Returns
    -------

    """
    machine_config = _load_config(config, json_path)
    if not machine_config.get("name"):
        raise MachineConfigException(
            "'%s' is empty" % join_json_paths("name", json_path)
        )
    return cast(MachineConfig, machine_config)


def load_model_config(config: dict, json_path: str = None) -> ModelConfig:
    """
    Load `ModelConfig` from the dict

    Parameters
    ----------
    config: str
        Config to load.
    json_path: str
        JSON path position of the config.

    Returns
    -------

    """
    model_config = cast(ModelConfig, load_machine_config(config, json_path))
    if not model_config.get("project_name"):
        raise MachineConfigException(
            "'%s' is empty" % join_json_paths("project_name", json_path)
        )
    return model_config
