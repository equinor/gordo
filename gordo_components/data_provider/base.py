# -*- coding: utf-8 -*-

import abc

import functools
import inspect

from datetime import datetime
from typing import Iterable, List, Callable, Optional

import pandas as pd

from gordo_components.dataset.sensor_tag import SensorTag


# TODO: Move this to a more appropriate module
def capture_args(method: Callable):
    """
    Decorator to capture ``args`` and ``kwargs`` passed to a given method
    Assumed there is a ``self`` to which it assigned the attribute to as ``_params``
    as one dict of kwargs

    Parameters
    ----------
    method: Callable
        Some method of an object, with 'self' as the first parameter

    Returns
    -------
    Any
        Returns whatever the original method would return
    """

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        """
        Need to map args to the parameters in the method and then update that
        param dict with explicitly provided kwargs and assign that to _params
        """
        params = {}
        for arg_val, arg_key in zip(
            args, (arg for arg in inspect.getfullargspec(method).args if arg != "self")
        ):
            params[arg_key] = arg_val
        params.update(kwargs)
        self._params = params
        return method(self, *args, **kwargs)

    return wrapper


class GordoBaseDataProvider(object):
    @abc.abstractmethod
    def load_series(
        self,
        from_ts: datetime,
        to_ts: datetime,
        tag_list: List[SensorTag],
        dry_run: Optional[bool] = False,
    ) -> Iterable[pd.Series]:
        """
        Load the required data as an iterable of series where each
        contains the values of the tag with time index

        Parameters
        ----------
        from_ts: datetime
            Datetime object representing the start of fetching data
        to_ts: datetime
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
    def __init__(self, **kwargs):
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
