# -*- coding: utf-8 -*-

from typing import List, Optional, Type, Any, cast
from copy import copy

from gordo.machine.validators import fix_runtime
from gordo.workflow.workflow_generator.helpers import patch_dict
from gordo.machine import (
    Machine,
    GlobalsConfig,
    load_machine_config,
    load_globals_config,
)
from gordo.utils import join_json_paths
from gordo import __version__
from gordo_core.import_utils import BackCompatibleLocations
from packaging.version import parse
from pydantic import parse_obj_as, BaseModel

from .schemas import BuilderPodRuntime, PodRuntime, Volume


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

    SPLITED_DOCKER_IMAGES: GlobalsConfig = {
        "runtime": {
            "deployer": {"image": "gordo-deploy"},
            "server": {"image": "gordo-model-server"},
            "prometheus_metrics_server": {"image": "gordo-model-server"},
            "builder": {"image": "gordo-model-builder"},
            "client": {"image": "gordo-client"},
        }
    }

    UNIFYING_GORDO_VERSION: str = "1.2.0"

    UNIFIED_DOCKER_IMAGES: GlobalsConfig = {
        "runtime": {
            "deployer": {"image": "gordo-base"},
            "server": {"image": "gordo-base"},
            "prometheus_metrics_server": {"image": "gordo-base"},
            "builder": {"image": "gordo-base"},
            "client": {"image": "gordo-base"},
        }
    }

    DEFAULT_CONFIG_GLOBALS: GlobalsConfig = {
        "runtime": {
            "reporters": [],
            "server": {
                "resources": {
                    "requests": {"memory": 3000, "cpu": 1000},
                    "limits": {"memory": 6000, "cpu": 2000},
                }
            },
            "prometheus_metrics_server": {
                "resources": {
                    "requests": {"memory": 200, "cpu": 100},
                    "limits": {"memory": 1000, "cpu": 200},
                }
            },
            "builder": {
                "resources": {
                    "requests": {"memory": 3900, "cpu": 1001},
                    "limits": {"memory": 31200, "cpu": 1001},
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

    def __init__(
        self,
        config: dict,
        project_name: str,
        gordo_version: Optional[str] = None,
        model_builder_env: Optional[dict] = None,
        back_compatibles: Optional[BackCompatibleLocations] = None,
        default_data_provider: Optional[str] = None,
        json_path: Optional[str] = None,
    ):
        if gordo_version is None:
            gordo_version = __version__
        default_globals = self.get_default_globals(gordo_version)
        default_globals["runtime"]["influx"][  # type: ignore
            "resources"
        ] = _calculate_influx_resources(  # type: ignore
            len(config["machines"])
        )

        passed_globals = load_globals_config(
            config.get("globals", dict()), join_json_paths("globals", json_path)
        )

        # keeping it for back-compatibility
        if model_builder_env is not None and not (
            default_globals
            and "runtime" in default_globals
            and "builder" in default_globals["runtime"]
            and "env" in default_globals["runtime"]["builder"]
        ):
            if "runtime" not in default_globals:
                default_globals["runtime"] = {}
            if "builder" not in default_globals["runtime"]:
                default_globals["runtime"]["builder"] = {}
            default_globals["runtime"]["builder"]["env"] = model_builder_env

        patched_globals = cast(
            GlobalsConfig,
            patch_dict(cast(dict, default_globals), cast(dict, passed_globals)),
        )
        patched_globals = self.prepare_patched_globals(patched_globals)

        self.project_name = project_name

        machines: list[Machine] = []
        for i, conf in enumerate(config["machines"]):
            machine_config = load_machine_config(
                conf, join_json_paths("machines[%d]" % i, json_path)
            )
            machine = Machine.from_config(
                cast(dict[str, Any], machine_config),
                project_name=project_name,
                config_globals=patched_globals,
                back_compatibles=back_compatibles,
                default_data_provider=default_data_provider,
            )
            machines.append(machine)

        self.machines = machines

        self.globals: GlobalsConfig = patched_globals

    @staticmethod
    def prepare_runtime(runtime: dict) -> dict:
        def prepare_pod_runtime(name: str, schema: Type[BaseModel] = PodRuntime):
            if name in runtime:
                # TODO handling pydantic.ValidationError
                pod_runtime = parse_obj_as(schema, runtime[name])
                runtime[name] = pod_runtime.dict(exclude_none=True)

        prepare_pod_runtime("builder", BuilderPodRuntime)
        if "volumes" in runtime:
            volumes = parse_obj_as(List[Volume], runtime["volumes"])
            runtime["volumes"] = [volume.dict(exclude_none=True) for volume in volumes]
        return runtime

    @classmethod
    def prepare_patched_globals(cls, patched_globals: GlobalsConfig) -> GlobalsConfig:
        runtime = fix_runtime(patched_globals.get("runtime"))
        runtime = cls.prepare_runtime(runtime)
        patched_globals["runtime"] = runtime
        return patched_globals

    @classmethod
    def get_default_globals(cls, gordo_version: str) -> GlobalsConfig:
        current_version = parse(gordo_version)
        unifying_version = parse(cls.UNIFYING_GORDO_VERSION)
        if current_version >= unifying_version:
            docker_images = cls.UNIFIED_DOCKER_IMAGES
        else:
            docker_images = cls.SPLITED_DOCKER_IMAGES
        default_globals = cls.DEFAULT_CONFIG_GLOBALS
        return cast(
            GlobalsConfig,
            patch_dict(cast(dict, copy(default_globals)), cast(dict, docker_images)),
        )
