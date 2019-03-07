# -*- coding: utf-8 -*-

import logging
import numpy as np
import pandas as pd
from datetime import datetime

from gordo_components.dataset.base import GordoBaseDataset
from gordo_components.data_provider.base import GordoBaseDataProvider

logger = logging.getLogger(__name__)


class TimeSeriesDataset(GordoBaseDataset):
    def __init__(
        self,
        data_provider: GordoBaseDataProvider,
        from_ts: datetime,
        to_ts: datetime,
        tag_list: list,
        resolution: str = "10T",
        **_kwargs,
    ):
        self.from_ts = from_ts
        self.to_ts = to_ts
        self.tag_list = tag_list
        self.resolution = resolution
        self.data_provider = data_provider

        if not self.from_ts.tzinfo or not self.to_ts.tzinfo:
            raise ValueError(
                f"Timestamps ({self.from_ts}, {self.to_ts}) need to include timezone "
                f"information"
            )

    def get_data(self) -> pd.DataFrame:
        dataframes = self.data_provider.load_dataframes(
            from_ts=self.from_ts, to_ts=self.to_ts, tag_list=self.tag_list
        )
        X = self.join_timeseries(dataframes, self.from_ts, self.resolution)
        y = None
        return X, y

    def get_metadata(self):
        metadata = {
            "tag_list": self.tag_list,
            "train_start_date": self.from_ts,
            "train_end_date": self.to_ts,
            "resolution": self.resolution,
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
