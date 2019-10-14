# -*- coding: utf-8 -*-

from typing import Dict, Any

from .base import ConfigElement


class DataProvider(ConfigElement):
    """
    Represents a DataProvider
    """

    def __init__(self, type: str = "DataLakeProvider", **kwargs):
        """
        Parameters
        ----------
        type: str
            Type of DataProvider to initialize
        kwargs: dict
            Other parameters which will be provided to the dataprovider on
            initialization
        """
        self.data_provider_type = type
        self.kwargs = kwargs

    def to_dict(self):
        config = {"type": self.data_provider_type}
        config.update(self.kwargs)  # Other optional kwargs, such as threads
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "DataProvider":
        return cls(**config)
