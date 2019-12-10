# -*- coding: utf-8 -*-

from typing import Dict, Any, List
from datetime import datetime

from .base import ConfigElement
from gordo_components.workflow.config_elements.validators import (
    ValidTagList,
    ValidDatetime,
    ValidDatasetKwargs,
)
from gordo_components.dataset.datasets import compat


class Dataset(ConfigElement):
    """
    Represents a dataset key element by machine within the config file
    and used to create a Dataset from components
    """

    # Required arguments for Dataset
    tag_list = ValidTagList()
    target_tag_list = ValidTagList()
    train_start_date = ValidDatetime()
    train_end_date = ValidDatetime()

    # Optional kwargs to constructing the dataset, such as 'resolution'
    kwargs = ValidDatasetKwargs()

    @compat
    def __init__(
        self,
        tag_list: List[str],
        target_tag_list: List[str],
        from_ts: datetime,
        to_ts: datetime,
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
        self.tag_list = tag_list
        self.target_tag_list = target_tag_list
        self.train_start_date = from_ts
        self.train_end_date = to_ts
        if self.train_start_date >= self.train_end_date:
            raise ValueError(
                f"from_ts ({self.train_start_date}) must be before "
                f"to_ts ({self.train_end_date})"
            )
        self.kwargs = kwargs

    def to_dict(self):
        config = {
            "type": "TimeSeriesDataset",
            "tag_list": self.tag_list,
            "target_tag_list": self.target_tag_list,
            "train_start_date": self.train_start_date.isoformat(),
            "train_end_date": self.train_end_date.isoformat(),
        }
        config.update(self.kwargs)  # Other optional kwargs, such as resolution.

        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Dataset":

        # Set target_tag_list as tags if it wasn't set; we always expect there to be target tags
        config = config.copy()
        config.setdefault("target_tag_list", config.get("tags") or config["tag_list"])

        return cls(**config)
