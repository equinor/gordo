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

    @abc.abstractmethod
    def __init__(self, **kwargs):
        ...

    @abc.abstractmethod
    def can_handle_tag(self, tag):
        """ Returns true if the dataprovider thinks it can possibly read this tag.

        Does not guarantee success, but is should be a pretty good guess
        (typically a regular expression is used to determine of the reader can read the
        tag)"""
        ...
