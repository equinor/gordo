# -*- coding: utf-8 -*-

import abc

from datetime import datetime
from typing import Iterable, List, Optional

import pandas as pd

from gordo.machine.dataset.sensor_tag import SensorTag


class GordoBaseDataProvider(object):
    @abc.abstractmethod
    def load_series(
        self,
        train_start_date: datetime,
        train_end_date: datetime,
        tag_list: List[SensorTag],
        dry_run: Optional[bool] = False,
    ) -> Iterable[pd.Series]:
        """
        Load the required data as an iterable of series where each
        contains the values of the tag with time index

        Parameters
        ----------
        train_start_date: datetime
            Datetime object representing the start of fetching data
        train_end_date: datetime
            Datetime object representing the end of fetching data
        tag_list: List[SensorTag]
            List of tags to fetch, where each will end up being its own dataframe
        dry_run: Optional[bool]
            Set to true to perform a "dry run" of the loading.
            Up to the implementations to determine what that means.

        Returns
        -------
        Iterable[pd.Series]
        """
        ...

    @abc.abstractmethod
    def can_handle_tag(self, tag: SensorTag):
        """
        Returns true if the dataprovider thinks it can possibly read this tag.
        Typically checks if the asset part of the tag is known to the reader.

        Parameters
        ----------
        tag: SensorTag - Dictionary with a "tag" key and optional "asset"

        Returns
        -------
        bool

        """
        ...

    @abc.abstractmethod
    def to_dict(self):
        """
        Serialize this object into a dict representation, which can be used to
        initialize a new object after popping 'type' from the dict.

        Returns
        -------
        dict
        """
        if not hasattr(self, "_params"):
            raise AttributeError(
                f"Failed to lookup init parameters, ensure the "
                f"object's __init__ is decorated with 'capture_args'"
            )
        # Update dict with the class
        params = self._params
        params["type"] = self.__class__.__name__
        return params

    @classmethod
    @abc.abstractmethod
    def from_dict(cls, config: dict) -> "GordoBaseDataProvider":
        from gordo.machine.dataset.data_provider import providers

        Provider = getattr(providers, config.get("type", "DataLakeProvider"))
        if Provider is None:
            raise TypeError(f"No data provider of type '{config['type']}'")
        return Provider(**config)
