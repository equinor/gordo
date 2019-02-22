# -*- coding: utf-8 -*-

import abc

from datetime import datetime
from typing import Iterable, List

import pandas as pd


class GordoBaseDataProvider(abc.ABC):
    @abc.abstractmethod
    def load_dataframes(
        self, from_ts: datetime, to_ts: datetime, tag_list: List[str]
    ) -> Iterable[pd.DataFrame]:
        """
        Load the required data as an iterable of dataframes where each
        is a single column dataframe with time index

        Parameters
        ----------
        from_ts: datetime - Datetime object representing the start of fetching data
        to_ts: datetime - Datetime object representing the end of fetching data
        tag_list: List[str] - List of tags to fetch, where each will end up being its own dataframe

        Returns
        -------
        Iterable[pd.DataFrame]
        """
        ...
