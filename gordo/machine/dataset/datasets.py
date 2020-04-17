# -*- coding: utf-8 -*-
import logging

from typing import Tuple, List, Dict, Union, Optional, Iterable, Callable, Sequence
from datetime import datetime
from dateutil.parser import isoparse
from functools import wraps

import pandas as pd
import numpy as np

from gordo.machine.dataset.data_provider.providers import (
    RandomDataProvider,
    DataLakeProvider,
)
from gordo.machine.dataset.base import GordoBaseDataset, InsufficientDataError
from gordo.machine.dataset.data_provider.base import GordoBaseDataProvider
from gordo.machine.dataset.filter_rows import pandas_filter_rows
from gordo.machine.dataset.sensor_tag import SensorTag
from gordo.machine.dataset.sensor_tag import normalize_sensor_tags
from gordo.util import capture_args
from gordo.machine.validators import (
    ValidTagList,
    ValidDatetime,
    ValidDatasetKwargs,
    ValidDataProvider,
)

logger = logging.getLogger(__name__)


class InsufficientDataAfterRowFilteringError(InsufficientDataError):
    pass


def compat(init):
    """
    __init__ decorator for compatibility where the Gordo config file's ``dataset`` keys have
    drifted from what kwargs are actually expected in the given dataset. For example,
    using `train_start_date` is common in the configs, but :class:`~TimeSeriesDataset`
    takes this parameter as ``train_start_date``, as well as :class:`~RandomDataset`

    Renames old/other acceptable kwargs to the ones that the dataset type expects
    """

    @wraps(init)
    def wrapper(*args, **kwargs):
        renamings = {
            "from_ts": "train_start_date",
            "to_ts": "train_end_date",
            "tags": "tag_list",
        }
        for old, new in renamings.items():
            if old in kwargs:
                kwargs[new] = kwargs.pop(old)
        return init(*args, **kwargs)

    return wrapper


class TimeSeriesDataset(GordoBaseDataset):

    train_start_date = ValidDatetime()
    train_end_date = ValidDatetime()
    tag_list = ValidTagList()
    target_tag_list = ValidTagList()
    data_provider = ValidDataProvider()
    kwargs = ValidDatasetKwargs()

    @compat
    @capture_args
    def __init__(
        self,
        train_start_date: Union[datetime, str],
        train_end_date: Union[datetime, str],
        tag_list: Sequence[Union[str, Dict, SensorTag]],
        target_tag_list: Optional[Sequence[Union[str, Dict, SensorTag]]] = None,
        data_provider: Union[GordoBaseDataProvider, dict] = DataLakeProvider(),
        resolution: Optional[str] = "10T",
        row_filter: str = "",
        aggregation_methods: Union[str, List[str], Callable] = "mean",
        row_filter_buffer_size: int = 0,
        asset: Optional[str] = None,
        default_asset: Optional[str] = None,
        n_samples_threshold: int = 0,
        **_kwargs,
    ):
        """
        Creates a TimeSeriesDataset backed by a provided dataprovider.

        A TimeSeriesDataset is a dataset backed by timeseries, but resampled,
        aligned, and (optionally) filtered.

        Parameters
        ----------
        train_start_date: Union[datetime, str]
            Earliest possible point in the dataset (inclusive)
        train_end_date: Union[datetime, str]
            Earliest possible point in the dataset (exclusive)
        tag_list: Sequence[Union[str, Dict, sensor_tag.SensorTag]]
            List of tags to include in the dataset. The elements can be strings,
            dictionaries or SensorTag namedtuples.
        target_tag_list: Sequence[List[Union[str, Dict, sensor_tag.SensorTag]]]
            List of tags to set as the dataset y. These will be treated the same as
            tag_list when fetching and pre-processing (resampling) but will be split
            into the y return from ``.get_data()``
        data_provider: Union[GordoBaseDataProvider, dict]
            A dataprovider which can provide dataframes for tags from train_start_date to train_end_date
            of which can also be a config definition from a data provider's ``.to_dict()`` method.
        resolution: Optional[str]
            The bucket size for grouping all incoming time data (e.g. "10T").
            Available strings come from https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
            **Note**: If this parameter is ``None`` or ``False``, then _no_ aggregation/resampling is applied to the data.
        row_filter: str
            Filter on the rows. Only rows satisfying the filter will be in the dataset.
            See :func:`gordo.machine.dataset.filter_rows.pandas_filter_rows` for
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
        n_samples_threshold: int = 0
            The threshold at which the generated DataFrame is considered to have too few rows of data.
        _kwargs
        """
        self.train_start_date = self._validate_dt(train_start_date)
        self.train_end_date = self._validate_dt(train_end_date)

        if self.train_start_date >= self.train_end_date:
            raise ValueError(
                f"train_end_date ({self.train_end_date}) must be after train_start_date ({self.train_start_date})"
            )

        self.tag_list = normalize_sensor_tags(list(tag_list), asset, default_asset)
        self.target_tag_list = (
            normalize_sensor_tags(list(target_tag_list), asset, default_asset)
            if target_tag_list
            else self.tag_list.copy()
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
        self.n_samples_threshold = n_samples_threshold

        if not self.train_start_date.tzinfo or not self.train_end_date.tzinfo:
            raise ValueError(
                f"Timestamps ({self.train_start_date}, {self.train_end_date}) need to include timezone "
                f"information"
            )

    def to_dict(self):
        params = super().to_dict()
        to_str = lambda dt: str(dt) if not hasattr(dt, "isoformat") else dt.isoformat()
        params["train_start_date"] = to_str(params["train_start_date"])
        params["train_end_date"] = to_str(params["train_end_date"])
        return params

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
            train_start_date=self.train_start_date,
            train_end_date=self.train_end_date,
            tag_list=list(set(self.tag_list + self.target_tag_list)),
        )

        # Resample if we have a resolution set, otherwise simply join the series.
        if self.resolution:
            data = self.join_timeseries(
                series_iter,
                self.train_start_date,
                self.train_end_date,
                self.resolution,
                aggregation_methods=self.aggregation_methods,
            )
        else:
            data = pd.concat(series_iter, axis=1, join="inner")

        if len(data) <= self.n_samples_threshold:
            raise InsufficientDataError(
                f"The length of the generated DataFrame ({len(data)}) does not exceed the "
                f"specified required threshold for number of rows ({self.n_samples_threshold})."
            )

        if self.row_filter:
            data = pandas_filter_rows(
                data, self.row_filter, buffer_size=self.row_filter_buffer_size
            )

            if len(data) <= self.n_samples_threshold:
                raise InsufficientDataAfterRowFilteringError(
                    f"The length of the genrated DataFrame ({len(data)}) does not exceed the "
                    f"specified required threshold for the number of rows ({self.n_samples_threshold}), "
                    f" after applying the specified row-filter."
                )

        x_tag_names = [tag.name for tag in self.tag_list]
        y_tag_names = [tag.name for tag in self.target_tag_list]

        X = data[x_tag_names]
        y = data[y_tag_names] if self.target_tag_list else None

        if X.first_valid_index():
            self._metadata["train_start_date_actual"] = X.index[0]
            self._metadata["train_end_date_actual"] = X.index[-1]

        self._metadata["summary_statistics"] = X.describe().to_dict()
        hists = dict()
        for tag in X.columns:
            step = round((X[tag].max() - X[tag].min()) / 100, 6)
            if step < 9e-07:
                hists[str(tag)] = "{}"
                continue
            outs = pd.cut(
                X[tag],
                bins=np.arange(
                    round(X[tag].min() - step, 6), round(X[tag].max() + step, 6), step,
                ),
                retbins=False,
            )
            hists[str(tag)] = outs.value_counts().sort_index().to_json(orient="index")
        self._metadata["x_hist"] = hists

        return X, y

    def get_metadata(self):
        return self._metadata.copy()


class RandomDataset(TimeSeriesDataset):
    """
    Get a TimeSeriesDataset backed by
    gordo.machine.dataset.data_provider.providers.RandomDataProvider
    """

    @compat
    @capture_args
    def __init__(
        self,
        train_start_date: Union[datetime, str],
        train_end_date: Union[datetime, str],
        tag_list: list,
        **kwargs,
    ):
        kwargs.pop("data_provider", None)  # Dont care what you ask for, you get random!
        super().__init__(
            data_provider=RandomDataProvider(),
            train_start_date=train_start_date,
            train_end_date=train_end_date,
            tag_list=tag_list,
            **kwargs,
        )
