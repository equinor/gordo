# -*- coding: utf-8 -*-

from typing import Dict, Any

import yaml

from gordo_components.workflow.config_elements.dataset import Dataset
from gordo_components.workflow.config_elements.validators import (
    ValidUrlString,
    ValidMetadata,
    ValidModel,
    ValidDataset,
)
from .base import ConfigElement


class Machine(ConfigElement):
    """
    Represents a single machine in a config file
    """

    name = ValidUrlString()
    model = ValidModel()
    dataset = ValidDataset()
    metadata = ValidMetadata()

    def __init__(
        self, name: str, model: dict, dataset: Dataset, metadata: dict = dict()
    ):
        self.name = name
        self.model = model
        self.dataset = dataset

        # Update metadata with the machine name
        metadata.update({"machine-name": name})
        self.metadata = metadata

    @classmethod
    def from_config(cls, config: Dict[str, Any], config_globals: dict = dict()):
        name = config["name"]
        model = config.get("model") or config_globals["model"]

        dataset = Dataset.from_config(
            config.get("dataset") or config_globals["dataset"]
        )
        metadata = {
            "global-metadata": config_globals.get("metadata", dict()),
            "machine-metadata": config.get("metadata", dict()),
        }
        return cls(name, model, dataset, metadata)

    def __str__(self):
        return yaml.dump(self.to_dict())

    def to_dict(self):
        return {
            "name": self.name,
            "dataset": self.dataset.to_dict(),
            "model": self.model,
            "metadata": self.metadata,
        }
