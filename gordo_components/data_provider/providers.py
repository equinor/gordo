# -*- coding: utf-8 -*-
import os
import re

from datetime import datetime
from typing import List, Iterable

import numpy as np
import pandas as pd
from azure.datalake.store import core, lib
from influxdb import DataFrameClient

from gordo_components.data_provider.base import GordoBaseDataProvider
from gordo_components.dataset.datasets import logger


class DataLakeProvider(GordoBaseDataProvider):
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
    ]

    def __init__(
        self,
        storename: str = "dataplatformdlsprod",
        interactive: bool = False,
        dl_service_auth_str: str = None,
    ):
        """
        Instantiates a DataLakeBackedDataset, for fetching of data from the data lake
        Parameters
        ----------
        storename: str - The store name to read data from
        interactive: bool - To perform authentication interactively, or attempt to do it automatically,
                            in such a case must provide 'del_service_authS_tr' parameter or as 'DL_SERVICE_AUTH_STR'
                            env var.
        dl_service_auth_str: Optional[str] - string on the format 'tenant_id:service_id:service_secret'. To perform
                                             authentication automatically; will default to DL_SERVICE_AUTH_STR env var or None

        """
        self.storename = storename
        self.interactive = interactive
        self.dl_service_auth_str = dl_service_auth_str or os.environ.get(
            "DL_SERVICE_AUTH_STR"
        )

    def load_dataframes(
        self, from_ts: datetime, to_ts: datetime, tag_list: List[str]
    ) -> Iterable[pd.DataFrame]:
        """
        See GordoBaseDataProvider for documentation
        """
        adls_file_system_client = self.create_adls_client()

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

    def read_tag_files(
        self, adls_file_system_client: core.AzureDLFileSystem, tag: str, years: range
    ) -> pd.DataFrame:
        """
        Download tag files for the given years into dataframes,
        and return as one dataframe.

        Parameters
        ----------
        adls_file_system_client: core.AzureDLFileSystem -
                                 the AzureDLFileSystem client to use
        tag: str - the tag to download data for
        years: range - range object providing years to include

        Returns
        -------
        pd.DataFrame: Single dataframe with all years for one tag.

        """
        tag_base_path = self.base_path_from_tag(tag)
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

        return combined

    def create_adls_client(self) -> core.AzureDLFileSystem:
        """
        Create ADLS file system client

        Returns
        -------
        core.AzureDLFileSystem: Instance of AzureDLFileSystem, ready for use
        """

        if self.interactive:
            token = self.get_datalake_token(interactive=True)

        else:
            token = self.get_datalake_token(
                interactive=False, dl_service_auth_str=self.dl_service_auth_str
            )

        adls_file_system_client = core.AzureDLFileSystem(
            token, store_name=self.storename
        )
        return adls_file_system_client

    @staticmethod
    def get_datalake_token(
        interactive=True, dl_service_auth_str=None
    ) -> lib.DataLakeCredential:
        logger.info("Looking for ways to authenticate with data lake")
        if interactive:
            return lib.auth()

        elif dl_service_auth_str:
            logger.info("Attempting to use datalake service authentication")
            dl_service_auth_elems = dl_service_auth_str.split(":")
            tenant = dl_service_auth_elems[0]
            client_id = dl_service_auth_elems[1]
            client_secret = dl_service_auth_elems[2]
            token = lib.auth(
                tenant_id=tenant,
                client_secret=client_secret,
                client_id=client_id,
                resource="https://datalake.azure.net/",
            )
            return token

        else:
            raise ValueError(
                f"Either interactive (value: {interactive}) must be True, "
                f"or dl_service_auth_str (value: {dl_service_auth_str}) "
                "must be set. "
            )

    def base_path_from_tag(self, tag):
        """

        :param tag:
        :return:
        """
        tag = tag.lower()
        logger.debug(f"Looking for pattern for tag {tag}")

        for pattern in self.TAG_TO_PATH:
            if pattern[0].match(tag):
                logger.info(
                    f"Found pattern {pattern[0]} in tag {tag}, returning {pattern[1]}"
                )
                return pattern[1]

        raise ValueError(f"Unable to find base path from tag {tag}")


class InfluxDataProvider(GordoBaseDataProvider):
    def __init__(
        self,
        measurement: str,
        api_key: str = None,
        api_key_header: str = None,
        value_name: str = "Value",
        **kwargs,
    ):
        """
        Parameters
        ----------
        measurement: str - Name of the measurement to select from in Influx
        value_name: str - Name of value to select, default to 'Value'
        api_key: str - Api key to use in header
        api_key_header: str - key of header to insert the api key for requests
        kwargs: dict - These are passed directly to the init args of influxdb.DataFrameClient
        """
        self.measurement = measurement
        self.value_name = value_name
        self.influx_config = kwargs
        self.influx_client = DataFrameClient(**kwargs)
        if api_key:
            if not api_key_header:
                raise ValueError(
                    "If supplying an api key, you must supply the header key to insert it under."
                )
            self.influx_client._headers[api_key_header] = api_key

    def load_dataframes(
        self, from_ts: datetime, to_ts: datetime, tag_list: List[str]
    ) -> Iterable[pd.DataFrame]:
        """
        See GordoBaseDataProvider for documentation
        """
        return (
            self.read_single_sensor(
                from_ts=from_ts, to_ts=to_ts, tag=tag, measurement=self.measurement
            )
            for tag in tag_list
        )

    def read_single_sensor(
        self, from_ts: datetime, to_ts: datetime, tag: str, measurement: str
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
            from_ts: datetime - Datetime to start querying for data
            to_ts: datetime - Datetime to stop query for data
            tag: str - Name of the tag to match in influx
            measurement: str - name of the measurement to select from
        Returns
        -------
            One column DataFrame
        """

        logger.info(f"Reading tag: {tag}")
        logger.info(f"Fetching data from {from_ts} to {to_ts}")
        query_string = f"""
            SELECT "{self.value_name}" as "{tag}" 
            FROM "{measurement}" 
            WHERE("tag" =~ /^{tag}$/) 
                {f"AND time >= {int(from_ts.timestamp())}s" if from_ts else ""} 
                {f"AND time <= {int(to_ts.timestamp())}s" if to_ts else ""}
        """

        logger.info(f"Query string: {query_string}")
        dataframes = self.influx_client.query(query_string)

        try:
            return list(dataframes.values())[0]

        except IndexError as e:
            list_of_tags = self._list_of_tags_from_influx()
            if tag not in list_of_tags:
                raise ValueError(f"tag {tag} is not found in influx")
            logger.error(
                f"Unable to find data for tag {tag} in the time range {from_ts} - {to_ts}"
            )
            raise e

    def _list_of_tags_from_influx(self):
        query_tags = (
            f"""SHOW TAG VALUES ON {self.influx_config["database"]} WITH KEY="tag" """
        )
        result = self.influx_client.query(query_tags)
        list_of_tags = []
        for item in list(result.get_points()):
            list_of_tags.append(item["value"])
        return list_of_tags
