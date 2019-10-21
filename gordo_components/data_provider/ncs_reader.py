# -*- coding: utf-8 -*-
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Iterable, List, Optional
from urllib.parse import quote

import numpy as np
import pandas as pd
from azure.datalake.store import core

from gordo_components.data_provider.base import GordoBaseDataProvider
from gordo_components.dataset.sensor_tag import SensorTag

logger = logging.getLogger(__name__)


class NcsReader(GordoBaseDataProvider):
    ASSET_TO_PATH = {
        "1101-sfb": "/raw/corporate/IMS Statfjord/sensordata/​1101-SFB",
        "1110-gfa": "/raw/corporate/Aspen MS - IP21 Gullfaks A/sensordata/1110-GFA",
        "1125-kvb": "/raw/corporate/PI System Operation Norway/sensordata/1125-KVB",
        "1130-troa": "/raw/corporate/Aspen MS - IP21 Troll A/sensordata/1130-TROA",
        "1138-val": "/raw/corporate/PI System Operation Norway/sensordata/1138-VAL",
        "1163-gdr": "/raw/corporate/PI System Manager Sleipner/sensordata/1163-GDR",
        "1180-nor": "/raw/corporate/PI System Operation North/sensordata/1180-NOR",
        "1190-asga": "/raw/corporate/PI System Operation North/sensordata/1190-ASGA",
        "1191-asgb": "/raw/corporate/PI System Operation North/sensordata/1191-ASGB",
        "1218-gkr": "/raw/corporate/PI System Manager Sleipner/sensordata/1218-GKR",
        "1230-vis": "/raw/corporate/Aspen MS - IP21 Visund/sensordata/1230-VIS",
        "1295-pera": "/raw/corporate/Aspen MS - IP21 Peregrino/sensordata/1295-PERA",
        "1755-gra": "/raw/corporate/Aspen MS - IP21 Grane/sensordata/1755-GRA",
        "1775-trob": "/raw/corporate/Aspen MS - IP21 Troll B/sensordata/1775-TROB",
        "1776-troc": "/raw/corporate/Aspen MS - IP21 Troll C/sensordata/1776-TROC",
        "1900-jsv": "/raw/corporate/PI System Operation Johan Sverdrup/sensordata/1900-JSV",
        "1901-jsv": "/raw/corporate/PI System Operation Johan Sverdrup/sensordata/1901-JSV",
        "1902-jsv": "/raw/corporate/PI System Operation Johan Sverdrup/sensordata/1902-JSV",
        "1903-jsv": "/raw/corporate/PI System Operation Johan Sverdrup/sensordata/1903-JSV",
    }

    def __init__(
        self,
        client: core.AzureDLFileSystem,
        threads: Optional[int] = 1,
        remove_status_codes: Optional[list] = [0],
        dl_base_path: Optional[str] = None,
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

        """
        super().__init__(**kwargs)
        self.client = client
        self.threads = threads
        self.remove_status_codes = remove_status_codes
        self.dl_base_path = dl_base_path
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
        from_ts: datetime,
        to_ts: datetime,
        tag_list: List[SensorTag],
        dry_run: Optional[bool] = False,
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
                    (tag_frame_all_years.index >= from_ts)
                    & (tag_frame_all_years.index < to_ts)
                ]
                yield filtered

    @staticmethod
    def read_tag_files(
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

        for year in years:
            tag_name_encoded = quote(tag.name, safe=" ")
            file_path = (
                f"{tag_base_path}/{tag_name_encoded}/{tag_name_encoded}_{year}.csv"
            )
            logger.info(f"Parsing file {file_path}")

            info = adls_file_system_client.info(file_path)
            file_size = info.get("length") / (1024 ** 2)
            logger.info(f"File size: {file_size:.2f}MB")
            if dry_run:
                logger.info("Dry run only, returning empty frame early")
                return pd.DataFrame()

            with adls_file_system_client.open(file_path, "rb") as f:
                df = pd.read_csv(
                    f,
                    sep=";",
                    header=None,
                    names=["Sensor", tag.name, "Timestamp", "Status"],
                    usecols=[tag.name, "Timestamp", "Status"],
                    dtype={tag.name: np.float32},
                    parse_dates=["Timestamp"],
                    date_parser=lambda col: pd.to_datetime(col, utc=True),
                    index_col="Timestamp",
                )

                df = df[~df["Status"].isin(remove_status_codes)]
                all_years.append(df)
                logger.info(f"Done parsing file {file_path}")

        combined = pd.concat(all_years)

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

        logger.info(f"Looking for match for asset {asset}")
        asset = asset.lower()
        if asset not in NcsReader.ASSET_TO_PATH:
            logger.info(
                f"Could not find match for asset {asset} in the list of "
                f"supported assets: {NcsReader.ASSET_TO_PATH.keys()}"
            )
            return None

        logger.info(
            f"Found asset code {asset}, returning {NcsReader.ASSET_TO_PATH[asset]}"
        )
        return NcsReader.ASSET_TO_PATH[asset]
