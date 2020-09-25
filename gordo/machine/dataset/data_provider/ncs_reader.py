# -*- coding: utf-8 -*-
import logging
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import timeit
import traceback
from typing import Iterable, List, Optional, cast
from urllib.parse import quote

import pandas as pd

from gordo.machine.dataset.data_provider.base import GordoBaseDataProvider
from gordo.machine.dataset.file_system.base import FileSystem
from gordo.machine.dataset.data_provider.file_type import (
    FileType,
    CsvFileType,
    ParquetFileType,
    TimeSeriesColumns,
)
from gordo.machine.dataset.sensor_tag import SensorTag
from gordo.util import capture_args

from .assets_config import AssetsConfig

logger = logging.getLogger(__name__)

time_series_columns = TimeSeriesColumns("Time", "Value", "Status")


class NcsFileLookup:
    def __init__(self, file_type: FileType):
        """
        Creates a files finder.

        Notes
        -----
        This implementation related only for :class:`gordo.machine.dataset.data_provider.ndc_reader.NcsReader` not for :class:`gordo.machine.dataset.data_provider.iroc_reader.IrocReader`

        Parameters
        ----------
        file_type : FileType
        """
        self.file_type = file_type

    def lookup(
        self, storage: FileSystem, dir_path: str, tag_name: str, year: int
    ) -> Optional[str]:
        """
        Find files with a given file type in Azure Data Lake

        Parameters
        ----------
        fs: FileSystem
            Azure Data Lake file system
        dir_path: str
            Base directory for finding files
        tag_name
            File tag (encoded version)
        year
            File year
        """
        raise NotImplementedError()


class NcsCsvLookup(NcsFileLookup):
    """
    Finder for CSV files
    """

    def __init__(self):
        header = ["Sensor", "Value", "Time", "Status"]
        super().__init__(CsvFileType(header, time_series_columns))

    def lookup(
        self, storage: FileSystem, dir_path: str, tag_name: str, year: int
    ) -> Optional[str]:
        file_extension = self.file_type.file_extension
        path = f"{dir_path}/{tag_name}_{year}{file_extension}"
        return path if storage.isfile(path) else None


class NcsParquetLookup(NcsFileLookup):
    """
    Finder for Parquet files
    """

    def __init__(self):
        super().__init__(ParquetFileType(time_series_columns))

    def lookup(
        self, storage: FileSystem, dir_path: str, tag_name: str, year: int
    ) -> Optional[str]:
        file_extension = self.file_type.file_extension
        path = f"{dir_path}/parquet/{tag_name}_{year}{file_extension}"
        return path if storage.isfile(path) else None


class NcsReader(GordoBaseDataProvider):

    ALL_FILE_LOOKUPS = OrderedDict(
        (("parquet", NcsParquetLookup()), ("csv", NcsCsvLookup()))
    )

    @classmethod
    def get_file_lookups(cls, lookup_for: List[str]) -> List[NcsFileLookup]:
        if not lookup_for:
            raise ValueError("'lookup_for' must not be empty")
        file_lookups = []
        for lookup_name in lookup_for:
            try:
                file_lookups.append(cls.ALL_FILE_LOOKUPS[lookup_name])
            except KeyError:
                raise ValueError(
                    "Wrong lookup type '%s' in 'lookup_for' property", lookup_name
                )
        return file_lookups

    @capture_args
    def __init__(
        self,
        storage: FileSystem,
        assets_config: AssetsConfig,
        threads: Optional[int] = 1,
        remove_status_codes: Optional[list] = [0, 64, 60, 8, 24, 3, 32768],
        dl_base_path: Optional[str] = None,
        lookup_for: Optional[List[str]] = None,
        storage_name: Optional[str] = None,
    ):
        """
        Creates a reader for tags from the Norwegian Continental Shelf. Currently
        only supports a small subset of assets.

        Parameters
        ----------
        storage: FileSystem
            Storage file system
        assets_config: AssetsConfig
            Assets config
        threads : Optional[int]
            Number of threads to use. If None then use 1 thread
        remove_status_codes: Optional[list]
            Removes data with Status code(s) in the list. By default it removes data
            with Status code 0.
        dl_base_path: Optional[str]
            Base bath used to override the asset to path dictionary. Useful for demos
            and other non-production settings.
        lookup_for:  Optional[List[str]]
            List of file finders by the file type name. Value by default: ``['parquet', 'csv']``
        storage_name: Optional[str]
            Used by ``AssetsConfig``

        Notes
        -----
        `lookup_for` provide list sorted by priority. It means that for value ``['csv', 'parquet']``
        the reader will prefer to find CSV files over Parquet

        """
        self.storage = storage
        self.assets_config = assets_config

        self.threads = threads
        self.remove_status_codes = remove_status_codes
        self.dl_base_path = dl_base_path

        if lookup_for is None:
            file_lookups = list(self.ALL_FILE_LOOKUPS.values())
        else:
            file_lookups = self.get_file_lookups(lookup_for)
        self.file_lookups = file_lookups
        if storage_name is None:
            storage_name = storage.name
        self.storage_name: str = storage_name
        logger.info(f"Starting NCS reader with {self.threads} threads")

    @property
    def reader_name(self) -> str:
        """
        Property used for validating result of `AssetsConfig.get_path()`
        """
        return "ncs_reader"

    def can_handle_tag(self, tag: SensorTag):
        """
        Implements GordoBaseDataProvider, see base class for documentation
        """
        return (
            self.dl_base_path is not None
            or self.base_path_from_asset(tag.asset) is not None
        )

    def load_series(
        self,
        train_start_date: datetime,
        train_end_date: datetime,
        tag_list: List[SensorTag],
        dry_run: Optional[bool] = False,
    ) -> Iterable[pd.Series]:
        """
        See GordoBaseDataProvider for documentation
        """
        if train_end_date < train_start_date:
            raise ValueError(
                f"NCS reader called with train_end_date: {train_end_date} before train_start_date: {train_start_date}"
            )
        storage = self.storage

        years = range(train_start_date.year, train_end_date.year + 1)

        with ThreadPoolExecutor(max_workers=self.threads) as executor:
            fetched_tags = executor.map(
                lambda tag: self.read_tag_files(
                    storage=storage,
                    tag=tag,
                    years=years,
                    dry_run=dry_run,
                    remove_status_codes=self.remove_status_codes,
                    dl_base_path=self.dl_base_path,
                ),
                tag_list,
            )

            for tag_frame_all_years in fetched_tags:
                filtered = tag_frame_all_years[
                    (tag_frame_all_years.index >= train_start_date)
                    & (tag_frame_all_years.index < train_end_date)
                ]
                yield filtered

    @staticmethod
    def _verify_tag_path_exist(fs: FileSystem, path: str):
        """
        Verify that the tag path exists, if not the `fs.info` will raise a FileNotFound error.

        Parameters
        ----------
        fs: FileSystem
            File system
        path : str
            Path of tag to be checked if exists.
        """
        fs.info(path)

    def read_tag_files(
        self,
        storage: FileSystem,
        tag: SensorTag,
        years: range,
        dry_run: Optional[bool] = False,
        remove_status_codes: Optional[list] = [0],
        dl_base_path: Optional[str] = None,
    ) -> pd.Series:
        """
        Download tag files for the given years into dataframes,
        and return as one dataframe.

        Parameters
        ----------
        storage: FileSystem
            File system
        tag: SensorTag
            the tag to download data for
        years: range
            range object providing years to include
        dry_run: Optional[bool]
            if True, don't download data, just check info, log, and return
        remove_status_codes: Optional[list]
            Removes data with Status code(s) in the list. By default it removes data
            with Status code 0.
        dl_base_path: Optional[str]
            Base bath used to override the asset to path dictionary. Useful for demos
            and other non-production settings.

        Returns
        -------
        pd.Series:
            Series with all years for one tag.
        """
        tag_base_path = (
            dl_base_path if dl_base_path else self.base_path_from_asset(tag.asset)
        )

        if not tag_base_path:
            raise ValueError(f"Unable to find base path from tag {tag} ")
        all_years = []
        logger.info(f"Downloading tag: {tag} for years: {years}")
        tag_name_encoded = quote(tag.name, safe=" ")

        NcsReader._verify_tag_path_exist(
            storage, f"{tag_base_path}/{tag_name_encoded}/"
        )

        dir_path = f"{tag_base_path}/{tag_name_encoded}"
        for year in years:
            file_path = ""
            file_lookup = None
            for v in self.file_lookups:
                lookup_file_path = v.lookup(storage, dir_path, tag_name_encoded, year)
                if lookup_file_path is not None:
                    file_path = cast(str, lookup_file_path)
                    file_lookup = v
                    break
            if file_lookup is None:
                continue
            file_type = file_lookup.file_type
            logger.info(f"Parsing file {file_path}")

            try:
                info = storage.info(file_path)
                file_size = info.size / (1024 ** 2)
                logger.debug(f"File size for file {file_path}: {file_size:.2f}MB")

                if dry_run:
                    logger.info("Dry run only, returning empty frame early")
                    return pd.Series()
                before_downloading = timeit.default_timer()
                with storage.open(file_path, "rb") as f:
                    df = file_type.read_df(f)
                    df = df.rename(columns={"Value": tag.name})
                    df = df[~df["Status"].isin(remove_status_codes)]
                    df.sort_index(inplace=True)
                    all_years.append(df)
                    logger.info(
                        f"Done in {(timeit.default_timer()-before_downloading):.2f} sec {file_path}"
                    )

            except FileNotFoundError as e:
                logger.debug(f"{file_path} not found, skipping it: {e}")

        try:
            combined = pd.concat(all_years)
        except Exception as e:
            logger.debug(f"Not able to concatinate all years: {e}.")
            return pd.Series(name=tag.name, data=None)

        # There often comes duplicated timestamps, keep the last
        if combined.index.duplicated().any():
            combined = combined[~combined.index.duplicated(keep="last")]

        return combined[tag.name]

    def base_path_from_asset(self, asset: str):
        """
        Resolves an asset code to the datalake basepath containing the data.
        Returns None if it does not match any of the asset codes we know.
        """
        if not asset:
            return None

        logger.debug(f"Looking for match for asset {asset}")
        asset = asset.lower()
        assets_config = self.assets_config
        path_spec = assets_config.get_path(self.storage_name, asset)
        if path_spec is None:
            return None
        if path_spec.reader != self.reader_name:
            return None
        full_path = path_spec.full_path(self.storage)

        logger.debug(f"Found asset code {asset}, returning {full_path}")
        return full_path
