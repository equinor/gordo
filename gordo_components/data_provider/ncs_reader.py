# -*- coding: utf-8 -*-
import logging
import re
from datetime import datetime
from typing import Iterable, List

import numpy as np
import pandas as pd
from azure.datalake.store import core

from gordo_components.data_provider.base import GordoBaseDataProvider

logger = logging.getLogger(__name__)


class NcsReader(GordoBaseDataProvider):
    TAG_TO_PATH = [
        (
            re.compile(r"^asgb."),
            "/raw/corporate/PI System Operation North/sensordata/1191-ASGB",
        ),
        (
            re.compile(r"^gra."),
            "/raw/corporate/Aspen MS - IP21 Grane/sensordata/1755-GRA",
        ),
        (
            re.compile(r"^1125."),
            "/raw/corporate/PI System Operation Norway/sensordata/1125-KVB",
        ),
        (
            re.compile(r"^trb."),
            "/raw/corporate/Aspen MS - IP21 Troll B/sensordata/1775-TROB",
        ),
        (
            re.compile(r"^trc."),
            "/raw/corporate/Aspen MS - IP21 Troll C/sensordata/1776-TROC",
        ),
        (
            re.compile(r"^tra."),
            "/raw/corporate/Aspen MS - IP21 Troll A/sensordata/1130-TROA",
        ),
    ]

    def __init__(self, client: core.AzureDLFileSystem, **kwargs):
        """
        Creates a reader for tags from the Norwegian Continental Shelf. Currently
        only supports a small subset of assets.

        """
        super().__init__(**kwargs)
        self.client = client

    def can_handle_tag(self, tag):
        return NcsReader.base_path_from_tag(tag)

    def load_series(
        self, from_ts: datetime, to_ts: datetime, tag_list: List[str]
    ) -> Iterable[pd.Series]:
        """
        See GordoBaseDataProvider for documentation
        """
        if to_ts < from_ts:
            raise ValueError(
                f"NCS reader called with to_ts: {to_ts} before from_ts: {from_ts}"
            )
        adls_file_system_client = self.client

        years = range(from_ts.year, to_ts.year + 1)

        for tag in tag_list:
            logger.info(f"Processing tag {tag}")

            tag_frame_all_years = self.read_tag_files(
                adls_file_system_client, tag, years
            )
            filtered = tag_frame_all_years[
                (tag_frame_all_years.index >= from_ts)
                & (tag_frame_all_years.index < to_ts)
            ]
            yield filtered

    @staticmethod
    def read_tag_files(
        adls_file_system_client: core.AzureDLFileSystem, tag: str, years: range
    ) -> pd.Series:
        """
        Download tag files for the given years into dataframes,
        and return as one dataframe.

        Parameters
        ----------
        adls_file_system_client: core.AzureDLFileSystem
            the AzureDLFileSystem client to use
        tag: str
            the tag to download data for
        years: range
            range object providing years to include

        Returns
        -------
        pd.Series:
            Series with all years for one tag.
        """
        tag_base_path = NcsReader.base_path_from_tag(tag)
        if not tag_base_path:
            raise ValueError(f"Unable to find base path from tag {tag}")
        all_years = []
        for year in years:
            file_path = tag_base_path + f"/{tag}/{tag}_{year}.csv"
            logger.info(f"Parsing file {file_path}")

            info = adls_file_system_client.info(file_path)
            file_size = info.get("length") / (1024 ** 2)
            logger.info(f"File size: {file_size:.2f}MB")

            with adls_file_system_client.open(file_path, "rb") as f:
                df = pd.read_csv(
                    f,
                    sep=";",
                    header=None,
                    names=["Sensor", tag, "Timestamp", "Status"],
                    usecols=[tag, "Timestamp"],
                    dtype={tag: np.float32},
                    parse_dates=["Timestamp"],
                    date_parser=lambda col: pd.to_datetime(col, utc=True),
                    index_col="Timestamp",
                )

                all_years.append(df)
                logger.info(f"Done parsing file {file_path}")

        combined = pd.concat(all_years)

        # There often comes duplicated timestamps, keep the last
        if combined.index.duplicated().any():
            combined = combined[~combined.index.duplicated(keep="last")]

        return combined[tag]

    @staticmethod
    def base_path_from_tag(tag):
        """
        Resolves a tag to the datalake basepath it should reside. Returns None if
        it does not match any of the tag-regexps we know.
        """
        tag = tag.lower()
        logger.debug(f"Looking for pattern for tag {tag}")

        for pattern in NcsReader.TAG_TO_PATH:
            if pattern[0].match(tag):
                logger.info(
                    f"Found pattern {pattern[0]} in tag {tag}, returning {pattern[1]}"
                )
                return pattern[1]
        return None
