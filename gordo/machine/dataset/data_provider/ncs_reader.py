# -*- coding: utf-8 -*-
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import timeit
from typing import Iterable, List, Optional, Tuple, cast

import pandas as pd

from gordo.machine.dataset.data_provider.base import GordoBaseDataProvider
from gordo.machine.dataset.file_system.base import FileSystem
from gordo.machine.dataset.sensor_tag import SensorTag
from gordo.util import capture_args

from .assets_config import AssetsConfig
from .ncs_contants import NCS_READER_NAME
from .ncs_file_type import load_ncs_file_types
from .ncs_lookup import NcsLookup, TagLocations

from ..exceptions import ConfigException

logger = logging.getLogger(__name__)


class NcsReader(GordoBaseDataProvider):

    DEFAULT_LOOKUP_FOR = ["parquet", "csv"]

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
        ncs_lookup: Optional[NcsLookup] = None,
        **kwargs,  # Do not remove this
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
        ncs_lookup: Optional[NcsLookup]
            Creates with current ``storage``, ``storage_name`` and ``lookup_for`` if None

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
            lookup_for = self.DEFAULT_LOOKUP_FOR
        self.lookup_for = lookup_for
        if storage_name is None:
            storage_name = storage.name
        self.storage_name: str = storage_name
        if ncs_lookup is None:
            ncs_lookup = self.create_ncs_lookup()
        elif isinstance(ncs_lookup, NcsLookup):
            raise ConfigException("ncs_lookup should be instance of NcsLookup")
        self.ncs_lookup = ncs_lookup
        logger.info(f"Starting NCS reader with {self.threads} threads")

    def create_ncs_lookup(self) -> NcsLookup:
        ncs_file_types = load_ncs_file_types(self.lookup_for)
        return NcsLookup(self.storage, ncs_file_types, self.storage_name)

    @property
    def reader_name(self) -> str:
        """
        Property used for validating result of `AssetsConfig.get_path()`
        """
        return NCS_READER_NAME

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

        years = list(range(train_start_date.year, train_end_date.year + 1))

        tag_dirs_iter = self.ncs_lookup.assets_config_tags_lookup(
            self.assets_config, tag_list, self.dl_base_path
        )

        with ThreadPoolExecutor(max_workers=self.threads) as executor:
            fetched_tags = executor.map(
                lambda tag_dirs: self._load_series_mapper(tag_dirs, years, dry_run),
                tag_dirs_iter,
            )

            for tag_frame_all_years in fetched_tags:
                filtered = tag_frame_all_years[
                    (tag_frame_all_years.index >= train_start_date)
                    & (tag_frame_all_years.index < train_end_date)
                ]
                yield filtered

    def _load_series_mapper(
        self,
        tag_dirs: Tuple[SensorTag, Optional[str]],
        years: List[int],
        dry_run: Optional[bool] = False,
    ) -> pd.Series:
        tag, tag_dir = tag_dirs
        if tag_dir is None:
            logger.info(
                "Unable to find tag '%s' (asset '%s') directory in storage '%s'",
                tag.name,
                tag.asset,
                self.storage_name,
            )
            return pd.Series()
        tag_locations = self.ncs_lookup.files_lookup(tag_dir, tag, years)
        return self.read_tag_locations(tag_locations, dry_run)

    def read_tag_locations(
        self, tag_locations: TagLocations, dry_run: Optional[bool] = False
    ) -> pd.Series:
        """
        Reads all data from files in ``tag_locations``

        Parameters
        ----------
        tag_locations: TagLocations
        dry_run: bool

        Returns
        -------
        pd.Series

        """
        tag = tag_locations.tag
        years = tag_locations.years()

        all_years = []
        logger.info(f"Downloading tag: {tag} for years: {years}")
        for tag, year, location in tag_locations:
            file_path = location.path
            file_type = location.file_type
            logger.info(f"Parsing file {file_path}")

            try:
                info = self.storage.info(file_path)
                file_size = info.size / (1024 ** 2)
                logger.debug(f"File size for file {file_path}: {file_size:.2f}MB")

                if dry_run:
                    logger.info("Dry run only, returning empty frame early")
                    return pd.Series()
                before_downloading = timeit.default_timer()
                with self.storage.open(file_path, "rb") as f:
                    df = file_type.read_df(f)
                    df = df.rename(columns={"Value": tag.name})
                    df = df[~df["Status"].isin(self.remove_status_codes)]
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
            return pd.Series(name=tag.name, data=[])

            # There often comes duplicated timestamps, keep the last
        if combined.index.duplicated().any():
            combined = combined[~combined.index.duplicated(keep="last")]

        return combined[tag.name]

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
        self, tag: SensorTag, years: range, dry_run: Optional[bool] = False,
    ) -> pd.Series:
        """
        Download tag files for the given years into dataframes,
        and return as one dataframe.

        Parameters
        ----------
        tag: SensorTag
            the tag to download data for
        years: range
            range object providing years to include
        dry_run: Optional[bool]
            if True, don't download data, just check info, log, and return

        Returns
        -------
        pd.Series:
            Series with all years for one tag.
        """
        tag_base_path = self.dl_base_path
        if not tag_base_path:
            tag_base_path = self.base_path_from_asset(tag.asset)

        if not tag_base_path:
            raise ValueError(f"Unable to find base path from tag {tag} ")
        logger.info(f"Downloading tag: {tag} for years: {years}")

        tag_dir = None
        for found_tag, dir_path in self.ncs_lookup.tag_dirs_lookup(
            tag_base_path, [tag]
        ):
            if found_tag == tag:
                tag_dir = dir_path
                break
        if tag_dir:
            raise FileNotFoundError(
                f"Unable to find location of {tag} in storage {self.storage_name}"
            )

        tag_locations = self.ncs_lookup.files_lookup(cast(str, tag_dir), tag, years)
        return self.read_tag_locations(tag_locations, dry_run)

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
            raise ValueError(
                "Assets reader name should be equal '%s' and not '%s'"
                % (self.reader_name, path_spec.reader)
            )
        full_path = path_spec.full_path(self.storage)

        logger.debug(f"Found asset code {asset}, returning {full_path}")
        return full_path
