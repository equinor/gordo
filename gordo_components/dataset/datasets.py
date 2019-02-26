# -*- coding: utf-8 -*-

import logging
from typing import Tuple, List

import numpy as np
import pandas as pd
from datetime import datetime

from gordo_components.dataset.base import GordoBaseDataset
from gordo_components.data_provider.base import GordoBaseDataProvider
from gordo_components.dataset.filter_rows import pandas_filter_rows

logger = logging.getLogger(__name__)


class TimeSeriesDataset(GordoBaseDataset):
    def __init__(
        self,
        data_provider: GordoBaseDataProvider,
        from_ts: datetime,
        to_ts: datetime,
        tag_list: List[str],
        resolution: str = "10T",
        row_filter: str = "",
        **_kwargs,
    ):
        """
        Creates a TimeSeriesDataset backed by a provided dataprovider.

        A TimeSeriesDataset is a dataset backed by timeseries, but resampled,
        aligned, and (optionally) filtered.

        Parameters
        ----------
        data_provider: GordoBaseDataProvider
            A dataprovider which can provide dataframes for tags from from_ts to to_ts
        from_ts: datetime
            Earliest possible point in the dataset (inclusive)
        to_ts: datetime
            Earliest possible point in the dataset (exclusive)
        tag_list: List[Str]
            List of tags to include in the dataset
        resolution: str
            The bucket size for grouping all incoming time data (e.g. "10T").
        row_filter: str
            Filter on the rows. Only rows satisfying the filter will be in the dataset.
            See :func:`gordo_components.dataset.filter_rows.pandas_filter_rows` for
            further documentation of the filter format.
        _kwargs
        """
        self.from_ts = from_ts
        self.to_ts = to_ts
        self.tag_list = tag_list
        self.resolution = resolution
        self.data_provider = data_provider
        self.row_filter = row_filter

        if not self.from_ts.tzinfo or not self.to_ts.tzinfo:
            raise ValueError(
                f"Timestamps ({self.from_ts}, {self.to_ts}) need to include timezone "
                f"information"
            )

    def get_data(self) -> Tuple[pd.DataFrame, None]:
        dataframes = self.data_provider.load_dataframes(
            from_ts=self.from_ts, to_ts=self.to_ts, tag_list=self.tag_list
        )
        X = self.join_timeseries(dataframes, self.from_ts, self.resolution)
        y = None
        if self.row_filter:
            X = pandas_filter_rows(X, self.row_filter)
        return X, y

    def get_metadata(self):
        metadata = {
            "tag_list": self.tag_list,
            "train_start_date": self.from_ts,
            "train_end_date": self.to_ts,
            "resolution": self.resolution,
            "filter": self.row_filter,
        }
        return metadata


class RandomDataset(GordoBaseDataset):
    """
    Get a GordoBaseDataset which returns random values for X and y
    """

    def __init__(self, size=100, n_features=20, **kwargs):
        self.size = size
        self.n_features = n_features

    def get_data(self):
        """return X and y data"""
        X = np.random.random(size=self.size * self.n_features).reshape(
            -1, self.n_features
        )
        return X, X.copy()

    def get_metadata(self):
        metadata = {"size": self.size, "n_features": self.n_features}
        return metadata
