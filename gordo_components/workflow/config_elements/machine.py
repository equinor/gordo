# -*- coding: utf-8 -*-

from typing import Dict, Any

import yaml
from gordo_components.workflow.config_elements.data_provider import DataProvider

from gordo_components.workflow.config_elements.dataset import Dataset
from gordo_components.workflow.config_elements.validators import (
    ValidUrlString,
    ValidMetadata,
    ValidModel,
    ValidDataset,
    ValidMachineRuntime,
    ValidDataProvider,
)
from gordo_components.workflow.workflow_generator.helpers import patch_dict
from .base import ConfigElement


class Machine(ConfigElement):
    """
    Represents a single machine in a config file
    """

    name = ValidUrlString()
    project_name = ValidUrlString()
    host = ValidUrlString()
    model = ValidModel()
    dataset = ValidDataset()
    data_provider = ValidDataProvider()
    metadata = ValidMetadata()
    runtime = ValidMachineRuntime()

    def __init__(
        self,
        name: str,
        model: dict,
        dataset: Dataset,
        data_provider: DataProvider,
        project_name: str,
        evaluation: dict,
        metadata=None,
        runtime=None,
    ):

        if runtime is None:
            runtime = dict()
        if metadata is None:
            metadata = dict()
        self.name = name
        self.model = model
        self.dataset = dataset
        self.data_provider = data_provider
        self.runtime = runtime
        self.evaluation = evaluation

        self.metadata = metadata
        self.project_name = project_name

        self.host = f"gordoserver-{self.project_name}-{self.name}"

    @classmethod
    def from_config(  # type: ignore
        cls, config: Dict[str, Any], project_name: str, config_globals=None
    ):
        if config_globals is None:
            config_globals = dict()
        else:
            config_globals = config_globals.copy()

        config = config.copy()

        name = config["name"]
        model = config.get("model") or config_globals.get("model")

        local_runtime = config.get("runtime", dict())
        runtime = patch_dict(config_globals.get("runtime", dict()), local_runtime)

        dataset_config = patch_dict(
            config.get("dataset", dict()), config_globals.get("dataset", dict())
        )
        dataset = Dataset.from_config(dataset_config)
        evaluation = patch_dict(
            config_globals.get("evaluation", dict()), config.get("evaluation", dict())
        )

        data_provider = DataProvider.from_config(
            patch_dict(
                config_globals.get("data_provider", dict()),
                config.get("data_provider", dict()),
            )
        )

        metadata = {
            "global-metadata": config_globals.get("metadata", dict()),
            "machine-metadata": config.get("metadata", dict()),
        }
        return cls(
            name,
            model,
            dataset,
            data_provider=data_provider,
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
            "metadata": self.metadata,
            "runtime": self.runtime,
            "data_provider": self.data_provider.to_dict(),
            "project_name": self.project_name,
            "evaluation": self.evaluation,
        }
