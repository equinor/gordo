# -*- coding: utf-8 -*-

from typing import List, Optional
from copy import copy

from gordo.machine.validators import fix_runtime
from gordo.workflow.workflow_generator.helpers import patch_dict
from gordo.machine import Machine
from gordo import __version__
from packaging.version import parse


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

    """
    Handles the conversion of a single Machine representation in config format
    and updates it with any features which are 'left out' inside of ``globals``
    key or the default config globals held here.
    """

    SPLITED_DOCKER_IMAGES = {
        "runtime": {
            "deployer": {"image": "gordo-deploy"},
            "server": {
                "image": "gordo-model-server",
            },
            "prometheus_metrics_server": {
                "image": "gordo-model-server",
            },
            "builder": {
                "image": "gordo-model-builder",
            },
            "client": {
                "image": "gordo-client",
            },
        }
    }

    UNIFYING_GORDO_VERSION = "1.2.0"

    UNIFIED_DOCKER_IMAGES = {
        "runtime": {
            "deployer": {"image": "gordo-base"},
            "server": {
                "image": "gordo-base",
            },
            "prometheus_metrics_server": {
                "image": "gordo-base",
            },
            "builder": {
                "image": "gordo-base",
            },
            "client": {
                "image": "gordo-base",
            },
        }
    }

    DEFAULT_CONFIG_GLOBALS = {
        "runtime": {
            "reporters": [],
            "server": {
                "resources": {
                    "requests": {"memory": 3000, "cpu": 1000},
                    "limits": {"memory": 6000, "cpu": 2000},
                },
            },
            "prometheus_metrics_server": {
                "resources": {
                    "requests": {"memory": 200, "cpu": 100},
                    "limits": {"memory": 1000, "cpu": 200},
                },
            },
            "builder": {
                "resources": {
                    "requests": {"memory": 3900, "cpu": 1001},
                    "limits": {"memory": 7800, "cpu": 1001},
                },
                "remote_logging": {"enable": False},
            },
            "client": {
                "resources": {
                    "requests": {"memory": 3500, "cpu": 100},
                    "limits": {"memory": 4000, "cpu": 2000},
                },
                "max_instances": 30,
            },
            "influx": {"enable": True},
        },
        "evaluation": {
            "cv_mode": "full_build",
            "scoring_scaler": "sklearn.preprocessing.MinMaxScaler",
            "metrics": [
                "explained_variance_score",
                "r2_score",
                "mean_squared_error",
                "mean_absolute_error",
            ],
        },
    }

    """
    Represents a fully loaded config file
    """

    def __init__(self, config: dict, project_name: str, gordo_version: Optional[str] = None):
        if gordo_version is None:
            gordo_version = __version__
        default_globals = self.get_default_globals(gordo_version)
        default_globals["runtime"]["influx"][  # type: ignore
            "resources"
        ] = _calculate_influx_resources(  # type: ignore
            len(config["machines"])
        )

        passed_globals = config.get("globals", dict())
        patched_globals = patch_dict(default_globals, passed_globals)
        if patched_globals.get("runtime"):
            patched_globals["runtime"] = fix_runtime(patched_globals.get("runtime"))
        self.project_name = project_name
        self.machines: List[Machine] = [
            Machine.from_config(
                conf, project_name=project_name, config_globals=patched_globals
            )
            for conf in config["machines"]
        ]  # type: List[Machine]

        self.globals: dict = patched_globals

    @classmethod
    def get_default_globals(cls, gordo_version: str) -> dict:
        current_version = parse(gordo_version)
        unifying_version = parse(cls.UNIFYING_GORDO_VERSION)
        if current_version >= unifying_version:
            docker_images = cls.UNIFIED_DOCKER_IMAGES 
        else:
            docker_images = cls.SPLITED_DOCKER_IMAGES
        default_globals = cls.DEFAULT_CONFIG_GLOBALS
        return patch_dict(copy(default_globals), docker_images)
