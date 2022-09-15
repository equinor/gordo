import json
import yaml

from typing import Any
from gordo_core.sensor_tag import SensorTag


class MachineJSONEncoder(json.JSONEncoder):

    def default(self, o: Any) -> Any:
        if isinstance(o, SensorTag):
            return o.to_json()
        return super().default(o)


class multiline_str(str):
    pass


class MachineSafeDumper(yaml.SafeDumper):
    pass


def _multiline_str_representer(dumper: yaml.SafeDumper, data: Any):
    return dumper.represent_scalar('tag:yaml.org,2002:str', data)


def _sensor_tag_representer(dumper: yaml.SafeDumper, data: Any):
    return dumper.represent_dict(data.to_json())


MachineSafeDumper.add_representer(multiline_str, _multiline_str_representer)
MachineSafeDumper.add_representer(SensorTag, _sensor_tag_representer)
