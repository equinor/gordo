# -*- coding: utf-8 -*-
import logging
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import timeit
from typing import Iterable, List, Optional
from urllib.parse import quote

import pandas as pd
from azure.datalake.store import core

from gordo.machine.dataset.data_provider.base import GordoBaseDataProvider
from gordo.machine.dataset.data_provider.file_type import (
    FileType,
    CsvFileType,
    ParquetFileType,
    TimeSeriesColumns,
)
from gordo.machine.dataset.data_provider.azure_utils import is_file
from gordo.machine.dataset.sensor_tag import SensorTag
from gordo.util import capture_args

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
        self, client: core.AzureDLFileSystem, dir_path: str, tag_name: str, year: int
    ) -> Optional[str]:
        """
        Find files with a given file type in Azure Data Lake

        Parameters
        ----------
        client: core.AzureDLFileSystem
            Azure Data Lake client
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
        self, client: core.AzureDLFileSystem, dir_path: str, tag_name: str, year: int
    ) -> Optional[str]:
        file_extension = self.file_type.file_extension
        path = f"{dir_path}/{tag_name}_{year}{file_extension}"
        return path if is_file(client, path) else None


class NcsParquetLookup(NcsFileLookup):
    """
    Finder for Parquet files
    """

    def __init__(self):
        super().__init__(ParquetFileType(time_series_columns))

    def lookup(
        self, client: core.AzureDLFileSystem, dir_path: str, tag_name: str, year: int
    ) -> Optional[str]:
        file_extension = self.file_type.file_extension
        path = f"{dir_path}/parquet/{tag_name}_{year}{file_extension}"
        return path if is_file(client, path) else None


class NcsReader(GordoBaseDataProvider):
    ASSET_TO_PATH = {
        # Paths on the datalake with problematic tag naming schemes (e.g. misplaced,
        # varying delimiters, non-unique, etc.) are commented with their assioted tag prefixes
        "1100-sfa": "/raw/corporate/IMS Statfjord/sensordata/1100-SFA",  # None
        "1101-sfb": "/raw/corporate/IMS Statfjord/sensordata/1101-SFB",  # None
        "1102-sfc": "/raw/corporate/IMS Statfjord/sensordata/1102-SFC",  # None
        "1110-gfa": "/raw/corporate/Aspen MS - IP21 Gullfaks A/sensordata/1110-GFA",
        "1111-gfb": "/raw/corporate/Aspen MS - IP21 Gullfaks B/sensordata/1111-GFB",
        "1112-gfc": "/raw/corporate/Aspen MS - IP21 Gullfaks C/sensordata/1112-GFC",
        "1120-vfr": "/raw/corporate/Aspen MS - IP21 Veslefrikk/sensordata/1120-VFR",
        "1125-kvb": "/raw/corporate/PI System Operation Norway/sensordata/1125-KVB",
        "1130-troa": "/raw/corporate/Aspen MS - IP21 Troll A/sensordata/1130-TROA",
        "1138-val": "/raw/corporate/PI System Operation Norway/sensordata/1138-VAL",
        "1140-sla": "/raw/corporate/PI System Manager Sleipner/sensordata/1140-SLA",  # None
        "1141-slt": "/raw/corporate/PI System Manager Sleipner/sensordata/1141-SLT",  # None
        "1142-slb": "/raw/corporate/PI System Manager Sleipner/sensordata/1142-SLB",  # None
        "1163-gdr": "/raw/corporate/PI System Manager Sleipner/sensordata/1163-GDR",  # None
        "1170-hd": "/raw/corporate/PI System Operation North/sensordata/1170-HD",
        "1175-kri": "/raw/corporate/PI System Operation North/sensordata/1175-KRI",  # kri.
        "1175-kris": "/raw/corporate/PI System Operation North/sensordata/1175-KRIS",  # kri.
        "1180-nor": "/raw/corporate/PI System Operation North/sensordata/1180-NOR",
        "1190-asga": "/raw/corporate/PI System Operation North/sensordata/1190-ASGA",
        "1191-asgb": "/raw/corporate/PI System Operation North/sensordata/1191-ASGB",  # asga. asgb-
        "1192-asgs": "/raw/corporate/PI System Operation North/sensordata/1192-ASGS",  # asgb.
        "1218-gkr": "/raw/corporate/PI System Manager Sleipner/sensordata/1218-GKR",  # None + 1218.
        "1219-aha": "/raw/corporate/PI System Operation Mam/sensordata/1219-AHA",
        "1220-sna": "/raw/corporate/IMS Snorre A/sensordata/1220-SNA",  # None
        "1221-snb": "/raw/corporate/IMS Snorre B/sensordata/1221-SNB",  # None
        "1230-vis": "/raw/corporate/Aspen MS - IP21 Visund/sensordata/1230-VIS",
        "1294-pera": "/raw/corporate/Aspen MS - IP21 Peregrino/sensordata/1294-PERA",  # per.
        "1295-pera": "/raw/corporate/Aspen MS - IP21 Peregrino/sensordata/1295-PERA",  # per.
        "1298-perb": "/raw/corporate/Aspen MS - IP21 Peregrino/sensordata/1298-PERB",  # per.
        "1299-per": "/raw/corporate/Aspen MS - IP21 Peregrino/sensordata/1299-PERF",  # per. keeping this for back-compatibility
        "1299-perf": "/raw/corporate/Aspen MS - IP21 Peregrino/sensordata/1299-PERF",  # per.
        "1340-met": "/raw/corporate/PI System Operation Norway/sensordata/1340-MET",
        "1380-sno": "/raw/corporate/Aspen MS - IP21 Hammerfest/sensordata/1380-SNO",  # 25haxxx_
        "1755-gra": "/raw/corporate/Aspen MS - IP21 Grane/sensordata/1755-GRA",
        "1760-hea": "/raw/corporate/PI System Operation Norway/sensordata/1760-HEA",
        "1765-osc": "/raw/corporate/Aspen MS - IP21 Oseberg C/sensordata/1765-OSC",
        "1766-oss": "/raw/corporate/Aspen MS - IP21 Oseberg South/sensordata/1766-OSS",
        "1767-ose": "/raw/corporate/Aspen MS - IP21 Oseberg East/sensordata/1767-OSE",
        "1772-osa": "/raw/corporate/Aspen MS - IP21 Oseberg Field Center/sensordata/1772-OSA",  # osf.
        "1774-osd": "/raw/corporate/Aspen MS - IP21 Oseberg Field Center/sensordata/1774-OSD",  # osf.
        "1775-trob": "/raw/corporate/Aspen MS - IP21 Troll B/sensordata/1775-TROB",
        "1776-troc": "/raw/corporate/Aspen MS - IP21 Troll C/sensordata/1776-TROC",
        "1886-mara": "/raw/corporate/PI System Operation Mam/sensordata/1886-MARA",
        "1900-jsv": "/raw/corporate/PI System Operation Johan Sverdrup/sensordata/1900-JSV",
        "1901-jsv": "/raw/corporate/PI System Operation Johan Sverdrup/sensordata/1901-JSV",
        "1902-jsv": "/raw/corporate/PI System Operation Johan Sverdrup/sensordata/1902-JSV",
        "1903-jsv": "/raw/corporate/PI System Operation Johan Sverdrup/sensordata/1903-JSV",
        "1904-jsv": "/raw/corporate/PI System Operation Johan Sverdrup/sensordata/1904-JSV",
    }

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
        client: core.AzureDLFileSystem,
        threads: Optional[int] = 1,
        remove_status_codes: Optional[list] = [0, 64, 60, 8, 24, 3, 32768],
        dl_base_path: Optional[str] = None,
        lookup_for: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Creates a reader for tags from the Norwegian Continental Shelf. Currently
        only supports a small subset of assets.

        Parameters
        ----------
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

        Notes
        -----
        `lookup_for` provide list sorted by priority. It means that for value ``['csv', 'parquet']``
        the reader will prefer to find CSV files over Parquet

        """
        self.client = client
        self.threads = threads
        self.remove_status_codes = remove_status_codes
        self.dl_base_path = dl_base_path

        if lookup_for is None:
            file_lookups = list(self.ALL_FILE_LOOKUPS.values())
        else:
            file_lookups = self.get_file_lookups(lookup_for)
        self.file_lookups = file_lookups
        logger.info(f"Starting NCS reader with {self.threads} threads")

    def can_handle_tag(self, tag: SensorTag):
        """
        Implements GordoBaseDataProvider, see base class for documentation
        """
        return (
            self.dl_base_path is not None
            or NcsReader.base_path_from_asset(tag.asset) is not None
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
        adls_file_system_client = self.client

        years = range(train_start_date.year, train_end_date.year + 1)

        with ThreadPoolExecutor(max_workers=self.threads) as executor:
            fetched_tags = executor.map(
                lambda tag: self.read_tag_files(
                    adls_file_system_client=adls_file_system_client,
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
    def _verify_tag_path_exist(
        adls_file_system_client: core.AzureDLFileSystem, path: str
    ):
        """
        Verify that the tag path exists, if not the `adls_file_system_client.info` will raise a FileNotFound error.

        Parameters
        ----------
        adls_file_system_client: core.AzureDLFileSystem
            the AzureDLFileSystem client to use
        path : str
            Path of tag to be checked if exists.
        """
        adls_file_system_client.info(f"{path}")

    def read_tag_files(
        self,
        adls_file_system_client: core.AzureDLFileSystem,
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
        adls_file_system_client: core.AzureDLFileSystem
            the AzureDLFileSystem client to use
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
            dl_base_path if dl_base_path else NcsReader.base_path_from_asset(tag.asset)
        )

        if not tag_base_path:
            raise ValueError(f"Unable to find base path from tag {tag} ")
        all_years = []
        logger.info(f"Downloading tag: {tag} for years: {years}")
        tag_name_encoded = quote(tag.name, safe=" ")

        NcsReader._verify_tag_path_exist(
            adls_file_system_client, f"{tag_base_path}/{tag_name_encoded}/"
        )

        dir_path = f"{tag_base_path}/{tag_name_encoded}"
        for year in years:
            file_path = None
            file_lookup = None
            for v in self.file_lookups:
                file_path = v.lookup(
                    adls_file_system_client, dir_path, tag_name_encoded, year
                )
                if file_path is not None:
                    file_lookup = v
                    break
            if file_lookup is None:
                continue
            file_type = file_lookup.file_type
            logger.info(f"Parsing file {file_path}")

            try:
                info = adls_file_system_client.info(file_path)
                file_size = info.get("length") / (1024 ** 2)
                logger.debug(f"File size for file {file_path}: {file_size:.2f}MB")

                if dry_run:
                    logger.info("Dry run only, returning empty frame early")
                    return pd.Series()
                before_downloading = timeit.default_timer()
                with adls_file_system_client.open(file_path, "rb") as f:
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

    @staticmethod
    def base_path_from_asset(asset: str):
        """
        Resolves an asset code to the datalake basepath containing the data.
        Returns None if it does not match any of the asset codes we know.
        """
        if not asset:
            return None

        logger.debug(f"Looking for match for asset {asset}")
        asset = asset.lower()
        if asset not in NcsReader.ASSET_TO_PATH:
            logger.warning(
                f"Could not find match for asset {asset} in the list of "
                f"supported assets: {NcsReader.ASSET_TO_PATH.keys()}"
            )
            return None

        logger.debug(
            f"Found asset code {asset}, returning {NcsReader.ASSET_TO_PATH[asset]}"
        )
        return NcsReader.ASSET_TO_PATH[asset]
