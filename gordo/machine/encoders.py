import json
import yaml

from typing import Any
from datetime import datetime
from gordo_core.sensor_tag import SensorTag

DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f+%z"


class MachineJSONEncoder(json.JSONEncoder):
    """
    A JSONEncoder for machine objects, handling datetime.datetime objects as strings
    and :class:`~gordo_core.sensor_tag.SensorTag` as a dict
    representation of a :class:`~gordo.machine.Machine`
    """

    def default(self, obj: Any) -> Any:
        if isinstance(obj, datetime):
            return obj.strftime(DATETIME_FORMAT)
        elif isinstance(obj, SensorTag):
            return obj.to_json()
        return super().default(obj)


class multiline_str(str):
    pass


class MachineSafeDumper(yaml.SafeDumper):
    pass


def _multiline_str_representer(dumper: yaml.SafeDumper, data: Any):
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")


def _datetime_representer(dumper: yaml.SafeDumper, data: Any):
    return dumper.represent_str(data.strftime(DATETIME_FORMAT))


def _sensor_tag_representer(dumper: yaml.SafeDumper, data: Any):
    return dumper.represent_dict(data.to_json())


MachineSafeDumper.add_representer(multiline_str, _multiline_str_representer)
MachineSafeDumper.add_representer(datetime, _datetime_representer)
MachineSafeDumper.add_representer(SensorTag, _sensor_tag_representer)
