# -*- coding: utf-8 -*-
import logging

from typing import Tuple, List, Dict, Union, Optional, Iterable, Callable, Sequence
from datetime import datetime
from dateutil.parser import isoparse
from functools import wraps

import pandas as pd

from gordo_components.data_provider.providers import (
    RandomDataProvider,
    DataLakeProvider,
)
from gordo_components.dataset.base import GordoBaseDataset
from gordo_components.data_provider.base import GordoBaseDataProvider
from gordo_components.dataset.filter_rows import pandas_filter_rows
from gordo_components.dataset.sensor_tag import SensorTag
from gordo_components.dataset.sensor_tag import normalize_sensor_tags

logger = logging.getLogger(__name__)


def compat(init):
    """
    __init__ decorator for compatibility where the Gordo config file's ``dataset`` keys have
    drifted from what kwargs are actually expected in the given dataset. For example,
    using `train_start_date` is common in the configs, but :class:`~TimeSeriesDataset`
    takes this parameter as ``from_ts``, as well as :class:`~RandomDataset`

    Renames old/other acceptable kwargs to the ones that the dataset type expects
    """

    @wraps(init)
    def wrapper(*args, **kwargs):
        renamings = {
            "train_start_date": "from_ts",
            "train_end_date": "to_ts",
            "tags": "tag_list",
        }
        for old, new in renamings.items():
            if old in kwargs:
                kwargs[new] = kwargs.pop(old)
        return init(*args, **kwargs)

    return wrapper


class TimeSeriesDataset(GordoBaseDataset):
    @compat
    def __init__(
        self,
        from_ts: Union[datetime, str],
        to_ts: Union[datetime, str],
        tag_list: Sequence[Union[str, Dict, SensorTag]],
        target_tag_list: Optional[Sequence[Union[str, Dict, SensorTag]]] = None,
        data_provider: Union[GordoBaseDataProvider, dict] = DataLakeProvider(),
        resolution: Optional[str] = "10T",
        row_filter: str = "",
        aggregation_methods: Union[str, List[str], Callable] = "mean",
        row_filter_buffer_size: int = 0,
        asset: Optional[str] = None,
        default_asset: Optional[str] = None,
        **_kwargs,
    ):
        """
        Creates a TimeSeriesDataset backed by a provided dataprovider.

        A TimeSeriesDataset is a dataset backed by timeseries, but resampled,
        aligned, and (optionally) filtered.

        Parameters
        ----------
        from_ts: Union[datetime, str]
            Earliest possible point in the dataset (inclusive)
        to_ts: Union[datetime, str]
            Earliest possible point in the dataset (exclusive)
        tag_list: Sequence[Union[str, Dict, sensor_tag.SensorTag]]
            List of tags to include in the dataset. The elements can be strings,
            dictionaries or SensorTag namedtuples.
        target_tag_list: Sequence[List[Union[str, Dict, sensor_tag.SensorTag]]]
            List of tags to set as the dataset y. These will be treated the same as
            tag_list when fetching and pre-processing (resampling) but will be split
            into the y return from ``.get_data()``
        data_provider: Union[GordoBaseDataProvider, dict]
            A dataprovider which can provide dataframes for tags from from_ts to to_ts
            of which can also be a config definition from a data provider's ``.to_dict()`` method.
        resolution: Optional[str]
            The bucket size for grouping all incoming time data (e.g. "10T").
            Available strings come from https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
            **Note**: If this parameter is ``None`` or ``False``, then _no_ aggregation/resampling is applied to the data.
        row_filter: str
            Filter on the rows. Only rows satisfying the filter will be in the dataset.
            See :func:`gordo_components.dataset.filter_rows.pandas_filter_rows` for
            further documentation of the filter format.
        aggregation_methods
            Aggregation method(s) to use for the resampled buckets. If a single
            resample method is provided then the resulting dataframe will have names
            identical to the names of the series it got in. If several
            aggregation-methods are provided then the resulting dataframe will
            have a multi-level column index, with the series-name as the first level,
            and the aggregation method as the second level.
            See :py:func::`pandas.core.resample.Resampler#aggregate` for more
            information on possible aggregation methods.
        row_filter_buffer_size: int
            Whatever elements are selected for removal based on the ``row_filter``, will also
            have this amount of elements removed fore and aft.
            Default is zero 0
        asset: Optional[str]
            Asset for which the tags are associated with.
        default_asset: Optional[str]
            Asset which will be used if `asset` is not provided and the tag is not
            resolvable to a specific asset.
        _kwargs
        """
        self.from_ts = self._validate_dt(from_ts)
        self.to_ts = self._validate_dt(to_ts)
        self.tag_list = normalize_sensor_tags(list(tag_list), asset, default_asset)
        self.target_tag_list = (
            normalize_sensor_tags(list(target_tag_list), asset, default_asset)
            if target_tag_list
            else []
        )
        self.resolution = resolution
        self.data_provider = (
            data_provider
            if not isinstance(data_provider, dict)
            else GordoBaseDataProvider.from_dict(data_provider)
        )
        self.row_filter = row_filter
        self.aggregation_methods = aggregation_methods
        self.row_filter_buffer_size = row_filter_buffer_size
        self.asset = asset

        if not self.from_ts.tzinfo or not self.to_ts.tzinfo:
            raise ValueError(
                f"Timestamps ({self.from_ts}, {self.to_ts}) need to include timezone "
                f"information"
            )

    @staticmethod
    def _validate_dt(dt: Union[str, datetime]) -> datetime:
        dt = dt if isinstance(dt, datetime) else isoparse(dt)
        if dt.tzinfo is None:
            raise ValueError(
                "Must provide an ISO formatted datetime string with timezone information"
            )
        return dt

    def get_data(self) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:

        series_iter: Iterable[pd.Series] = self.data_provider.load_series(
            from_ts=self.from_ts,
            to_ts=self.to_ts,
            tag_list=list(set(self.tag_list + self.target_tag_list)),
        )

        # Resample if we have a resolution set, otherwise simply join the series.
        if self.resolution:
            data = self.join_timeseries(
                series_iter,
                self.from_ts,
                self.to_ts,
                self.resolution,
                aggregation_methods=self.aggregation_methods,
            )
        else:
            data = pd.concat(series_iter, axis=1, join="inner")

        if self.row_filter:
            data = pandas_filter_rows(
                data, self.row_filter, buffer_size=self.row_filter_buffer_size
            )

        x_tag_names = [tag.name for tag in self.tag_list]
        y_tag_names = [tag.name for tag in self.target_tag_list]

        X = data[x_tag_names]
        y = data[y_tag_names] if self.target_tag_list else None

        return X, y

    def get_metadata(self):
        metadata = {
            "tag_list": self.tag_list,
            "target_tag_list": self.target_tag_list,
            "train_start_date": self.from_ts,
            "train_end_date": self.to_ts,
            "resolution": self.resolution,
            "filter": self.row_filter,
            "row_filter_buffer_size": self.row_filter_buffer_size,
            "data_provider": self.data_provider.to_dict(),
            "asset": self.asset,
        }
        return metadata


class RandomDataset(TimeSeriesDataset):
    """
    Get a TimeSeriesDataset backed by
    gordo_components.data_provider.providers.RandomDataProvider
    """

    @compat
    def __init__(self, from_ts: datetime, to_ts: datetime, tag_list: list, **kwargs):
        kwargs.pop("data_provider", None)  # Dont care what you ask for, you get random!
        super().__init__(
            data_provider=RandomDataProvider(),
            from_ts=from_ts,
            to_ts=to_ts,
            tag_list=tag_list,
            **kwargs,
        )
