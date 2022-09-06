import yaml

from copy import copy
from typing import TypedDict, Optional, Union

from gordo_core.exceptions import ConfigException

from gordo.machine.metadata import Metadata
from gordo_core.base import GordoBaseDataset


class MachineConfigException(ConfigException):
    pass


class _BaseConfig(TypedDict, total=False):
    model: dict
    dataset: Union[dict, GordoBaseDataset]
    evaluation: Optional[dict]
    metadata: Optional[Union[dict, Metadata]]
    runtime: Optional[dict]


class GlobalsConfig(_BaseConfig):
    pass


class MachineConfig(_BaseConfig):
    name: str
    project_name: str


_FIELDS = ("model", "dataset", "evaluation", "metadata", "runtime")


def _load_config(config: dict, json_path: str = None) -> dict:
    new_config = copy(config)
    for field in _FIELDS:
        val = config.get(field)
        if type(val) is str:
            try:
                new_val = yaml.safe_load(val)
            except yaml.YAMLError as e:
                json_paths = []
                if json_path is not None:
                    json_paths.append(json_path)
                json_paths.append(field)
                message = "Error loading YAML from '%s'" % ".".join(json_paths)
                raise MachineConfigException(message + ": " + str(e))
        else:
            new_val = val
        new_config[field] = new_val
    return new_config


def load_machine_config(config: dict, json_path: str = None) -> MachineConfig:
    machine_config = _load_config(config, json_path)
    if machine_config.get("name") is None:
        message = ""
        if json_path is None:
            message += "in '%s'" % json_path
        raise MachineConfigException("name is empty" + json_path)
    return machine_config


def load_global_config(config: dict, json_path: str = None) -> GlobalsConfig:
    return _load_config(config, json_path)
