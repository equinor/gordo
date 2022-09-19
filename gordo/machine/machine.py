# -*- coding: utf-8 -*-
import json
import logging
from typing import Any, Optional, List, Callable, cast
from copy import copy

import yaml

from gordo_core.base import GordoBaseDataset
from gordo_core.sensor_tag import SensorTag
from gordo_core.import_utils import BackCompatibleLocations
from gordo.machine.validators import (
    ValidUrlString,
    ValidMetadata,
    ValidModel,
    ValidDataset,
    ValidMachineRuntime,
)
from gordo.machine.metadata import Metadata
from gordo.workflow.workflow_generator.helpers import patch_dict
from gordo.utils import normalize_sensor_tags, TagsList

from .constants import MACHINE_YAML_FIELDS
from .loader import ModelConfig, GlobalsConfig
from .encoders import MachineJSONEncoder, MachineSafeDumper, multiline_str

logger = logging.getLogger(__name__)


class Machine:
    """
    Represents a single machine in a config file
    """

    name = ValidUrlString()
    project_name = ValidUrlString()
    host = ValidUrlString()
    model = ValidModel()
    dataset = ValidDataset()
    metadata = ValidMetadata()
    runtime = ValidMachineRuntime()
    _strict = True

    @staticmethod
    def prepare_evaluation(evaluation: Optional[dict]) -> dict:
        if evaluation is None:
            evaluation = dict(cv_mode="full_build")
        return evaluation

    def __init__(
        self,
        name: str,
        model: dict,
        dataset: GordoBaseDataset,
        project_name: str,
        evaluation: Optional[dict] = None,
        metadata: Optional[Metadata] = None,
        runtime: Optional[dict] = None,
    ):

        if runtime is None:
            runtime = dict()
        if metadata is None:
            metadata = cast(Any, Metadata).from_dict({})
        self.name = name
        self.model = model
        self.dataset = dataset
        self.runtime = runtime
        self.evaluation = self.prepare_evaluation(evaluation)
        self.metadata = metadata
        self.project_name = project_name

        # host validation
        self.host = f"gordoserver-{self.project_name}-{self.name}"

    # TODO TypedDict for config argument
    @classmethod
    def from_config(  # type: ignore
        cls,
        config: dict[str, Any],
        project_name: Optional[str] = None,
        config_globals: GlobalsConfig = None,
        back_compatibles: Optional[BackCompatibleLocations] = None,
        default_data_provider: Optional[str] = None,
    ):
        """
        Construct an instance from a block of YAML config file which represents
        a single Machine; loaded as a ``dict``.

        Parameters
        ----------
        config: dict[str, Any]
            The loaded block of config which represents a 'Machine' in YAML
        project_name: str
            Name of the project this Machine belongs to.
        config_globals:
            The block of config within the YAML file within `globals`
        back_compatibles: Optional[BackCompatibleLocations]
            See `gordo_core.import_utils.prepare_back_compatible_locations()` function for reference.
        default_data_provider: Optional[str]

        Returns
        -------
        :class:`~Machine`
        """
        if config_globals is None:
            config_globals = dict()

        name = config["name"]
        config_model = config.get("model") or config_globals.get("model")
        if config_model is None:
            raise ValueError("model is empty")
        model = cast(dict, config_model)

        if project_name is None:
            project_name = config.get("project_name", None)
        if project_name is None:
            raise ValueError("project_name is empty")

        local_runtime = cast(dict, config.get("runtime", dict()))
        runtime = patch_dict(
            cast(dict, config_globals.get("runtime", dict())), local_runtime
        )

        dataset = patch_dict(
            config.get("dataset", dict()), config_globals.get("dataset", dict())
        )
        config_evaluation = cls.prepare_evaluation(config.get("evaluation"))
        evaluation = patch_dict(
            cast(dict, config_globals.get("evaluation", dict())), config_evaluation
        )

        metadata = Metadata(
            user_defined={
                "global-metadata": config_globals.get("metadata", dict()),
                "machine-metadata": config.get("metadata", dict()),
            }
        )
        return cls.from_dict(
            {
                "name": name,
                "model": model,
                "dataset": dataset,
                "project_name": project_name,
                "evaluation": evaluation,
                "metadata": metadata,
                "runtime": runtime,
            },
            back_compatibles=back_compatibles,
            default_data_provider=default_data_provider,
        )

    def normalize_sensor_tags(self, tag_list: TagsList) -> List[SensorTag]:
        """
        Finding assets for all of the tags according to information from the dataset metadata

        Parameters
        ----------
        tag_list: TagsList

        Returns
        -------
        List[SensorTag]

        """
        metadata = self.metadata
        build_dataset_metadata = metadata.build_metadata.dataset.to_dict()
        asset: Optional[str] = None
        if hasattr(self.dataset, "asset"):
            asset = self.dataset.asset
        return normalize_sensor_tags(build_dataset_metadata, tag_list, asset=asset)

    def __str__(self):
        return self.to_yaml()

    def __eq__(self, other):
        return self.to_dict() == other.to_dict()

    # TODO TypedDict for d argument
    @classmethod
    def from_dict(
        cls,
        d: ModelConfig,
        back_compatibles: Optional[BackCompatibleLocations] = None,
        default_data_provider: Optional[str] = None,
    ) -> "Machine":
        """
        Create
        A dict taken from either gordo config or :func:`~Machine.to_dict`.
        """
        # No special treatment required, just here for consistency.
        d = copy(d)
        if "dataset" in d and isinstance(d["dataset"], dict):
            d["dataset"] = GordoBaseDataset.from_dict(
                d["dataset"],
                back_compatibles=back_compatibles,
                default_data_provider=default_data_provider,
            )
        if "metadata" in d and isinstance(d["metadata"], dict):
            d["metadata"] = cast(Any, Metadata).from_dict(d["metadata"])
        args = cast(dict, d)
        return cls(**args)

    def to_dict(self):
        """
        Convert to a ``dict`` representation along with all attributes which
        can also be converted to a ``dict``. Can reload with :func:`~Machine.from_dict`
        """
        return {
            "name": self.name,
            "dataset": self.dataset.to_dict(),
            "model": self.model,
            "metadata": self.metadata.to_dict(),
            "runtime": self.runtime,
            "project_name": self.project_name,
            "evaluation": self.evaluation,
        }

    def _to_yaml_dict(self, yaml_serializer: Callable[[Any], str] = None):
        if yaml_serializer is None:
            raise ValueError("yaml_serializer is empty")
        machine = self.to_dict()
        config = {}
        for k, v in machine.items():
            if k in MACHINE_YAML_FIELDS:
                v = yaml_serializer(v)
            config[k] = v
        return config

    def to_json(self):
        """
        Returns
        -------
            string JSON representation of the machine.
        """
        json_dumps: Callable[[Any], Any] = lambda v: json.dumps(
            v, cls=MachineJSONEncoder
        )
        return json_dumps(self._to_yaml_dict(json_dumps))

    def to_yaml(self):
        """
        Returns
        -------
            string YAML representation of the machine.
        """
        yaml_dump: Callable[[Any], Any] = lambda v: multiline_str(
            yaml.dump(v, Dumper=MachineSafeDumper)
        )
        return yaml.dump(self._to_yaml_dict(yaml_dump), Dumper=MachineSafeDumper)

    def report(self):
        """
        Run any reporters in the machine's runtime for the current state.

        Reporters implement the :class:`gordo.reporters.base.BaseReporter` and
        can be specified in a config file of the machine for example:

        .. code-block:: yaml

            runtime:
              reporters:
                - gordo.reporters.postgres.PostgresReporter:
                    host: my-special-host

        """
        # Avoid circular dependency with reporters which import Machine
        from gordo.reporters.base import BaseReporter

        for reporter in map(BaseReporter.from_dict, self.runtime.get("reporters", [])):
            logger.debug(f"Using reporter: {reporter}")
            reporter.report(self)
