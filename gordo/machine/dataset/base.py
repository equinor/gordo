# -*- coding: utf-8 -*-

import abc
import logging
from typing import Iterable, Union, List, Callable, Dict, Any, Tuple
from datetime import datetime
from copy import copy

import pandas as pd
import numpy as np


logger = logging.getLogger(__name__)


class InsufficientDataError(ValueError):
    pass


class GordoBaseDataset:

    _params: Dict[Any, Any] = dict()  # provided by @capture_args on child's __init__
    _metadata: Dict[Any, Any] = dict()

    @abc.abstractmethod
    def get_data(
        self,
    ) -> Tuple[Union[np.ndarray, pd.DataFrame], Union[np.ndarray, pd.DataFrame]]:
        """
        Return X, y data as numpy or pandas' dataframes given current state
        """

    @abc.abstractmethod
    def to_dict(self) -> dict:
        """
        Serialize this object into a dict representation, which can be used to
        initialize a new object using :func:`~GordoBaseDataset.from_dict`

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
        Construct the dataset using a config from :func:`~GordoBaseDataset.to_dict`
        """
        from gordo.machine.dataset import datasets

        config = copy(config)
        Dataset = getattr(datasets, config.pop("type", "TimeSeriesDataset"))
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
        Get metadata about the current state of the dataset
        """

    def join_timeseries(
        self,
        series_iterable: Iterable[pd.Series],
        resampling_startpoint: datetime,
        resampling_endpoint: datetime,
        resolution: str,
        aggregation_methods: Union[str, List[str], Callable] = "mean",
        interpolation_method: str = "linear_interpolation",
        interpolation_limit: str = "8H",
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
        interpolation_method: str
            How should missing values be interpolated. Either forward fill (`ffill`) or by linear
            interpolation (default, `linear_interpolation`).
        interpolation_limit: str
            Parameter sets how long from last valid data point values will be interpolated/forward filled.
            Default is eight hours (`8H`).
            If None, all missing values are interpolated/forward filled.

        Returns
        -------
        pd.DataFrame
            A dataframe without NaNs, a common time index, and one column per
            element in the dataframe_generator. If multiple aggregation methods
            are provided then the resulting dataframe will have a multi-level column
            index with series-names as top-level and aggregation-method as second-level.

        """
        resampled_series = []
        missing_data_series = []

        key = "tag_loading_metadata"
        self._metadata[key] = dict()

        for series in series_iterable:
            self._metadata[key][series.name] = dict(original_length=len(series))
            try:
                resampled = GordoBaseDataset._resample(
                    series,
                    resampling_startpoint=resampling_startpoint,
                    resampling_endpoint=resampling_endpoint,
                    resolution=resolution,
                    aggregation_methods=aggregation_methods,
                    interpolation_method=interpolation_method,
                    interpolation_limit=interpolation_limit,
                )
            except IndexError:
                missing_data_series.append(series.name)
            else:
                resampled_series.append(resampled)
                self._metadata[key][series.name].update(
                    dict(resampled_length=len(resampled))
                )
        if missing_data_series:
            raise InsufficientDataError(
                f"The following features are missing data: {missing_data_series}"
            )

        joined_df = pd.concat(resampled_series, axis=1, join="inner")

        # Before returning, delete all rows with NaN, they were introduced by the
        # insertion of NaNs in the beginning of all timeseries
        dropped_na = joined_df.dropna()

        self._metadata[key]["aggregate_metadata"] = dict(
            joined_length=len(joined_df), dropped_na_length=len(dropped_na)
        )
        return dropped_na

    @staticmethod
    def _resample(
        series: pd.Series,
        resampling_startpoint: datetime,
        resampling_endpoint: datetime,
        resolution: str,
        aggregation_methods: Union[str, List[str], Callable] = "mean",
        interpolation_method: str = "linear_interpolation",
        interpolation_limit: str = "8H",
    ):
        """
        Takes a single series and resamples it.
        See :class:`gordo.machine.dataset.base.GordoBaseDataset.join_timeseries`
        """

        startpoint_sametz = resampling_startpoint.astimezone(tz=series.index[0].tzinfo)
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
            endpoint = pd.Series([np.NaN], index=[endpoint_sametz], name=series.name)
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

        resampled = series.resample(resolution, label="left").agg(aggregation_methods)
        # If several aggregation methods are provided, agg returns a dataframe
        # instead of a series. In this dataframe the column names are the
        # aggregation methods, like "max" and "mean", so we have to make a
        # multi-index with the series-name as the top-level and the
        # aggregation-method as the lower-level index.
        # For backwards-compatibility we *dont* return a multi-level index
        # when we have a single resampling method.
        if isinstance(resampled, pd.DataFrame):  # Several aggregation methods provided
            resampled.columns = pd.MultiIndex.from_product(
                [[series.name], resampled.columns], names=["tag", "aggregation_method"]
            )

        if interpolation_method not in ["linear_interpolation", "ffill"]:
            raise ValueError(
                "Interpolation method should be either linear_interpolation of ffill"
            )

        if interpolation_limit is not None:
            limit = int(
                pd.Timedelta(interpolation_limit).total_seconds()
                / pd.Timedelta(resolution).total_seconds()
            )

            if limit <= 0:
                raise ValueError(
                    "Interpolation limit must be larger than given resolution"
                )

        if interpolation_method == "linear_interpolation":
            return resampled.interpolate(limit=limit).dropna()

        else:
            return resampled.fillna(method=interpolation_method, limit=limit).dropna()
