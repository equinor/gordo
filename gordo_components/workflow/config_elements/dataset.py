# -*- coding: utf-8 -*-

from typing import Dict, Any, List
from datetime import datetime

from .base import ConfigElement
from gordo_components.workflow.config_elements.validators import (
    ValidTagList,
    ValidDatetime,
    ValidDatasetKwargs,
)


class Dataset(ConfigElement):
    """
    Represents a dataset key element by machine within the config file
    and used to create a Dataset from components
    """

    # Required arguments for Dataset
    tags = ValidTagList()
    target_tag_list = ValidTagList()
    train_start_date = ValidDatetime()
    train_end_date = ValidDatetime()

    # Optional kwargs to constructing the dataset, such as 'resolution'
    kwargs = ValidDatasetKwargs()

    def __init__(
        self,
        tags: List[str],
        target_tag_list: List[str],
        train_start_date: datetime,
        train_end_date: datetime,
        **kwargs,
    ):
        """
        Parameters
        ----------
        tags: List[str]
            Tags the dataset consists of.
        train_start_date: datetime
            Earliest possible date (inclusive) of the dataset
        train_end_date: datetime
            Latest possible date of the dataset, must be after `train_start_date`
        kwargs

        """
        self.tags = tags
        self.target_tag_list = target_tag_list
        self.train_start_date = train_start_date
        self.train_end_date = train_end_date
        if train_start_date >= train_end_date:
            raise ValueError(
                f"train_start_date ({train_start_date}) must be before "
                f"train_end_date ({train_end_date})"
            )
        self.kwargs = kwargs

    def to_dict(self):
        config = {
            "type": "TimeSeriesDataset",
            "tags": self.tags,
            "target_tag_list": self.target_tag_list,
            "train_start_date": self.train_start_date.isoformat(),
            "train_end_date": self.train_end_date.isoformat(),
        }
        config.update(self.kwargs)  # Other optional kwargs, such as resolution.

        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Dataset":

        # Set target_tag_list as tags if it wasn't set; we always expect there to be target tags
        config.setdefault("target_tag_list", config["tags"])

        return cls(**config)
