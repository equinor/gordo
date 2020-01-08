# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, Optional

import yaml

from gordo.machine.dataset.base import GordoBaseDataset
from gordo.machine.validators import (
    ValidUrlString,
    ValidMetadata,
    ValidModel,
    ValidDataset,
    ValidMachineRuntime,
)
from gordo.machine.metadata import Metadata
from gordo.workflow.workflow_generator.helpers import patch_dict


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

    def __init__(
        self,
        name: str,
        model: dict,
        dataset: Union[GordoBaseDataset, dict],
        project_name: str,
        evaluation: Optional[dict] = None,
        metadata: Optional[Union[dict, Metadata]] = None,
        runtime=None,
    ):

        if runtime is None:
            runtime = dict()
        if evaluation is None:
            evaluation = dict(cv_mode="full_build")
        if metadata is None:
            metadata = dict()
        self.name = name
        self.model = model
        self.dataset = (
            dataset
            if isinstance(dataset, GordoBaseDataset)
            else GordoBaseDataset.from_dict(dataset)
        )
        self.runtime = runtime
        self.evaluation = evaluation
        self.metadata = (
            metadata
            if isinstance(metadata, Metadata)
            else Metadata.from_dict(metadata)  # type: ignore
        )
        self.project_name = project_name

        self.host = f"gordoserver-{self.project_name}-{self.name}"

    @classmethod
    def from_config(  # type: ignore
        cls, config: Dict[str, Any], project_name: str, config_globals=None
    ):
        if config_globals is None:
            config_globals = dict()

        name = config["name"]
        model = config.get("model") or config_globals.get("model")

        local_runtime = config.get("runtime", dict())
        runtime = patch_dict(config_globals.get("runtime", dict()), local_runtime)

        dataset_config = patch_dict(
            config.get("dataset", dict()), config_globals.get("dataset", dict())
        )
        dataset = GordoBaseDataset.from_dict(dataset_config)
        evaluation = patch_dict(
            config_globals.get("evaluation", dict()), config.get("evaluation", dict())
        )

        metadata = Metadata(
            user_defined={
                "global-metadata": config_globals.get("metadata", dict()),
                "machine-metadata": config.get("metadata", dict()),
            }
        )
        return cls(
            name,
            model,
            dataset,
            metadata=metadata,
            runtime=runtime,
            project_name=project_name,
            evaluation=evaluation,
        )

    def __str__(self):
        return yaml.dump(self.to_dict())

    def __eq__(self, other):
        return self.to_dict() == other.to_dict()

    def to_dict(self):
        return {
            "name": self.name,
            "dataset": self.dataset.to_dict(),
            "model": self.model,
            "metadata": self.metadata.to_dict(),
            "runtime": self.runtime,
            "project_name": self.project_name,
            "evaluation": self.evaluation,
        }
