# -*- coding: utf-8 -*-

import abc
import logging
from typing import Iterable, Union, List, Callable, Dict, Any
from datetime import datetime

import pandas as pd
import numpy as np


logger = logging.getLogger(__name__)


class GordoBaseDataset:

    _params: Dict[Any, Any] = dict()  # provided by @capture_args on child's __init__

    @abc.abstractmethod
    def get_data(self):
        """
        Using initialized params, returns X, y as numpy arrays.
        """

    @abc.abstractmethod
    def to_dict(self) -> dict:
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
        for key, value in params.items():
            if hasattr(value, "to_dict"):
                params[key] = value.to_dict()
        return params

    @classmethod
    @abc.abstractmethod
    def from_dict(cls, config: Dict[str, Any]) -> "GordoBaseDataset":
        """
        Construct the dataset from a config without doing any validation
        """
        from gordo.machine.dataset import datasets

        Dataset = getattr(datasets, config.get("type", "TimeSeriesDataset"))
        if Dataset is None:
            raise TypeError(f"No dataset of type '{config['type']}'")

        # TODO: Here for compatibility, but @compate should take care of it, remove later
        if "tags" in config:
            config["tag_list"] = config.pop("tags")
        config.setdefault("target_tag_list", config["tag_list"])
        return Dataset(**config)

    @abc.abstractmethod
    def get_metadata(self):
        """
        Return metadata about the dataset in primitive / json encode-able dict form.
        """

    @staticmethod
    def join_timeseries(
        series_iterable: Iterable[pd.Series],
        resampling_startpoint: datetime,
        resampling_endpoint: datetime,
        resolution: str,
        aggregation_methods: Union[str, List[str], Callable] = "mean",
    ) -> pd.DataFrame:
        """

        Parameters
        ----------
        series_iterable: Iterable[pd.Series]
            An iterator supplying series with time index
        resampling_startpoint: datetime.datetime
            The starting point for resampling. Most data frames will not have this
            in their datetime index, and it will be inserted with a NaN as the value.
            The resulting NaNs will be removed, so the only important requirement for this is
            that this resampling_startpoint datetime must be before or equal to the first
            (earliest) datetime in the data to be resampled.
        resampling_endpoint: datetime.datetime
            The end point for resampling. This datetime must be equal to or after the last datetime in the
            data to be resampled.
        resolution: str
            The bucket size for grouping all incoming time data (e.g. "10T")
            Available strings come from https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
        aggregation_methods: Union[str, List[str], Callable]
            Aggregation method(s) to use for the resampled buckets. If a single
            resample method is provided then the resulting dataframe will have names
            identical to the names of the series it got in. If several
            aggregation-methods are provided then the resulting dataframe will
            have a multi-level column index, with the series-name as the first level,
            and the aggregation method as the second level.
            See :py:func::`pandas.core.resample.Resampler#aggregate` for more
            information on possible aggregation methods.

        Returns
        -------
        pd.DataFrame
            A dataframe without NaNs, a common time index, and one column per
            element in the dataframe_generator. If multiple aggregation methods
            are provided then the resulting dataframe will have a multi-level column
            index with series-names as top-level and aggregation-method as second-level.

        """
        resampled_series = []

        for series in series_iterable:
            startpoint_sametz = resampling_startpoint.astimezone(
                tz=series.index[0].tzinfo
            )
            endpoint_sametz = resampling_endpoint.astimezone(tz=series.index[0].tzinfo)

            if series.index[0] > startpoint_sametz:
                # Insert a NaN at the startpoint, to make sure that all resampled
                # indexes are the same. This approach will "pad" most frames with
                # NaNs, that will be removed at the end.
                startpoint = pd.Series(
                    [np.NaN], index=[startpoint_sametz], name=series.name
                )
                series = startpoint.append(series)
                logging.debug(
                    f"Appending NaN to {series.name} " f"at time {startpoint_sametz}"
                )

            elif series.index[0] < resampling_startpoint:
                msg = (
                    f"Error - for {series.name}, first timestamp "
                    f"{series.index[0]} is before the resampling start point "
                    f"{startpoint_sametz}"
                )
                logging.error(msg)
                raise RuntimeError(msg)

            if series.index[-1] < endpoint_sametz:
                endpoint = pd.Series(
                    [np.NaN], index=[endpoint_sametz], name=series.name
                )
                series = series.append(endpoint)
                logging.debug(
                    f"Appending NaN to {series.name} " f"at time {endpoint_sametz}"
                )
            elif series.index[-1] > endpoint_sametz:
                msg = (
                    f"Error - for {series.name}, last timestamp "
                    f"{series.index[-1]} is later than the resampling end point "
                    f"{endpoint_sametz}"
                )
                logging.error(msg)
                raise RuntimeError(msg)

            logging.debug("Head (3) and tail(3) of dataframe to be resampled:")
            logging.debug(series.head(3))
            logging.debug(series.tail(3))

            resampled = series.resample(resolution, label="left").agg(
                aggregation_methods
            )
            # If several aggregation methods are provided, agg returns a dataframe
            # instead of a series. In this dataframe the column names are the
            # aggregation methods, like "max" and "mean", so we have to make a
            # multi-index with the series-name as the top-level and the
            # aggregation-method as the lower-level index.
            # For backwards-compatibility we *dont* return a multi-level index
            # when we have a single resampling method.
            if isinstance(
                resampled, pd.DataFrame
            ):  # Several aggregation methods provided
                resampled.columns = pd.MultiIndex.from_product(
                    [[series.name], resampled.columns],
                    names=["tag", "aggregation_method"],
                )
            filled = resampled.fillna(method="ffill")
            resampled_series.append(filled)

        new_series = pd.concat(resampled_series, axis=1, join="inner")
        # Before returning, delete all rows with NaN, they were introduced by the
        # insertion of NaNs in the beginning of all timeseries

        return new_series.dropna()
