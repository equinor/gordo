# -*- coding: utf-8 -*-
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from dateutil import tz
from typing import Iterable, List, Optional, cast

import pandas as pd

from gordo.machine.dataset.data_provider.base import GordoBaseDataProvider
from gordo.machine.dataset.file_system.base import FileSystem
from gordo.machine.dataset.sensor_tag import SensorTag
from gordo.machine.dataset.sensor_tag import to_list_of_strings
from gordo.util import capture_args

from .assets_config import AssetsConfig

logger = logging.getLogger(__name__)


class IrocReader(GordoBaseDataProvider):
    def can_handle_tag(self, tag: SensorTag):
        return self.base_path_from_asset(tag.asset) is not None

    @capture_args
    def __init__(
        self,
        storage: Optional[FileSystem],
        assets_config: Optional[AssetsConfig],
        threads: int = 50,
        storage_name: Optional[str] = None,
    ):
        """
        Creates a reader for tags from IROC.
        """
        self.storage = storage
        self.assets_config = assets_config
        self.threads = threads
        if self.threads is None:
            self.threads = 50
        if storage_name is None and storage is not None:
            storage_name = storage.name
        self.storage_name = storage_name
        logger.info(f"Starting IROC reader with {self.threads} threads")

    @property
    def reader_name(self) -> str:
        """
        Property used for validating result of `AssetsConfig.get_path()`
        """
        return "iroc_reader"

    def load_series(
        self,
        train_start_date: datetime,
        train_end_date: datetime,
        tag_list: List[SensorTag],
        dry_run: Optional[bool] = False,
    ):
        """
        See GordoBaseDataProvider for documentation
        """
        if dry_run:
            raise NotImplementedError("Dry run for IrocReader is not implemented")
        if not tag_list:
            logger.warning("Iroc reader called with empty tag_list, returning none")
            return
        if train_end_date < train_start_date:
            raise ValueError(
                f"Iroc reader called with train_end_date: {train_end_date} before train_start_date: {train_start_date}"
            )

        base_paths_from_assets = list(
            map(lambda tag: IrocReader.base_path_from_asset(tag.asset), tag_list)
        )
        if len(set(base_paths_from_assets)) != 1:
            raise ValueError(
                "Iroc reader found either more than one asset or no asset from the tag list provided"
            )
        elif None in base_paths_from_assets:
            raise ValueError("Iroc reader could not associate some tags to an asset.")

        base_path = base_paths_from_assets[0]

        # We query with an extra day on both sides since the way the files are
        # organized in the datalake does not account for timezones, so some timestamps
        # are in the wrong files

        all_base_paths = (
            f"{base_path}/{t.year:0>4d}/{t.month:0>2d}/{t.day:0>2d}/"
            for t in pd.date_range(
                start=train_start_date.astimezone(tz.tzutc()) - pd.Timedelta("1D"),
                end=train_end_date.astimezone(tz.tzutc()) + pd.Timedelta("1D"),
                freq="D",
            )
        )

        fetched_tags = self._fetch_all_iroc_files_from_paths(
            all_base_paths, train_start_date, train_end_date, tag_list
        )
        if len(fetched_tags) < 0:
            raise ValueError(
                f"Found no data for tags {tag_list} in the daterange {train_start_date} to "
                f"{train_end_date}"
            )

        concatted = pd.concat(fetched_tags, copy=False)

        if len(concatted.columns) != len(tag_list):
            raise ValueError(
                f"Did not find data for all tags, the missing tags are "
                f"{set(to_list_of_strings(tag_list))-set(concatted.columns)}"
            )

        for col in concatted.columns:
            withouth_na = concatted[col].dropna()
            withouth_na.sort_index(inplace=True)
            yield withouth_na

    def _fetch_all_iroc_files_from_paths(
        self,
        all_base_paths: Iterable[str],
        train_start_date: datetime,
        train_end_date: datetime,
        tag_list: List[SensorTag],
    ):
        # Generator over all files in all of the base_paths
        def _all_files():
            if self.storage is not None:
                for b_path in all_base_paths:
                    for f in self.storage.walk(b_path):
                        yield f

        with ThreadPoolExecutor(max_workers=self.threads) as executor:
            # Pandas.concat makes the generator into a list anyway, so no extra memory
            # is used here
            fetched_tags = list(
                executor.map(
                    lambda file_path: self._read_iroc_df_from_azure(
                        file_path=file_path,
                        train_start_date=train_start_date,
                        train_end_date=train_end_date,
                        tag_list=tag_list,
                    ),
                    _all_files(),
                )
            )
            fetched_tags = [tags for tags in fetched_tags if not tags is None]

        return fetched_tags

    def _read_iroc_df_from_azure(
        self, file_path, train_start_date: datetime, train_end_date: datetime, tag_list
    ):
        storage = self.storage
        if storage is None:
            return None

        logger.info("Attempting to open IROC file {}".format(file_path))

        try:
            with storage.open(file_path, "rb") as f:
                logger.info("Parsing file {}".format(file_path))
                df = read_iroc_file(f, train_start_date, train_end_date, tag_list)
            return df
        except:
            logger.warning(f"Problem parsing file {file_path}, skipping.")
            return None

    def base_path_from_asset(self, asset: str):
        """
        Resolves an asset code to the datalake basepath containing the data.
        Returns None if it does not match any of the asset codes we know.
        """
        if not asset:
            return None
        if self.assets_config is None:
            return None

        logger.debug(f"Looking for match for asset {asset}")
        asset = asset.lower()
        assets_config = self.assets_config
        path_spec = cast(AssetsConfig, assets_config).get_path(
            cast(str, self.storage_name), asset
        )
        if path_spec is None:
            return None
        if path_spec.reader != self.reader_name:
            return None
        full_path = path_spec.full_path(cast(FileSystem, self.storage))

        logger.debug(f"Found asset code {asset}, returning {full_path}")
        return full_path


def read_iroc_file(
    file_obj,
    train_start_date: datetime,
    train_end_date: datetime,
    tag_list: List[SensorTag],
) -> pd.DataFrame:
    """
    Reads a single iroc timeseries csv, and returns it as a pandas.DataFrame.
    The returned dataframe has timestamps as a DateTimeIndex, and upto one column
    per tag in tag_list, but excluding any tags which does not exist in the csv.

    Parameters
    ----------
    file_obj: str or path object or file-like object
        File object to read iroc timeseries data from
    train_start_date
        Only keep timestamps later or equal than this
    train_end_date
        Only keep timestamps earlier than this
    tag_list
        Only keep tags in this list.

    Returns
    -------
    pd.DataFrame
        Dataframe with timestamps as a DateTimeIndex, and up to one column
        per tag in tag_list, but excluding any tags which does not exist in the
        csv.

    """
    df = pd.read_csv(file_obj, sep=",", usecols=["tag", "value", "timestamp"])
    df = df[df["tag"].isin(to_list_of_strings(tag_list))]
    # Note, there are some "digital" sensors with string values,
    # now they are just NaN converted
    df["value"] = df["value"].apply(pd.to_numeric, errors="coerce", downcast="float")
    df.dropna(inplace=True, subset=["value"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.pivot(index="timestamp", columns="tag")
    df = df[(df.index >= train_start_date) & (df.index < train_end_date)]
    df.columns = df.columns.droplevel(0)
    return df
