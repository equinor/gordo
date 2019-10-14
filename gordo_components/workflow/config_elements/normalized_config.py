# -*- coding: utf-8 -*-

from typing import List

from gordo_components.workflow.config_elements.validators import fix_runtime
from gordo_components.workflow.workflow_generator.helpers import patch_dict
from .machine import Machine


def _calculate_influx_resources(nr_of_machines):
    return {
        "requests": {
            # The requests must be limited to keep the machine schedulable
            "memory": min(3000 + (220 * nr_of_machines), 28000),  # Between 3G and 28G
            "cpu": min(500 + (10 * nr_of_machines), 4000),  # Between 500m and 4000m
        },
        "limits": {
            "memory": min(3000 + (220 * nr_of_machines), 48000),
            "cpu": 10000 + (20 * nr_of_machines),
        },
    }


class NormalizedConfig:

    DEFAULT_RUNTIME_GLOBALS = {
        "runtime": {
            "server": {
                "resources": {
                    "requests": {"memory": 3000, "cpu": 1000},
                    "limits": {"memory": 6000, "cpu": 2000},
                }
            },
            "builder": {
                "resources": {
                    "requests": {"memory": 1000, "cpu": 500},
                    "limits": {"memory": 3000, "cpu": 32000},
                }
            },
            "client": {
                "resources": {
                    "requests": {"memory": 3500, "cpu": 100},
                    "limits": {"memory": 4000, "cpu": 2000},
                },
                "max_instances": 30,
            },
            "influx": {"enable": True},
        }
    }

    """
    Represents a fully loaded config file
    """

    machines: List[Machine]
    globals: dict

    def __init__(self, config: dict, project_name: str):
        default_globals = self.DEFAULT_RUNTIME_GLOBALS
        default_globals["runtime"]["influx"][  # type: ignore
            "resources"
        ] = _calculate_influx_resources(  # type: ignore
            len(config["machines"])
        )

        passed_globals = config.get("globals", dict())
        patched_globals = patch_dict(default_globals, passed_globals)
        if patched_globals.get("runtime"):
            patched_globals["runtime"] = fix_runtime(patched_globals.get("runtime"))

        self.machines = [
            Machine.from_config(
                conf, project_name=project_name, config_globals=patched_globals
            )
            for conf in config["machines"]
        ]  # type: List[Machine]

        self.globals = patched_globals
