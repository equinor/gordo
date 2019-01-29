# -*- coding: utf-8 -*-

import os
import re
import logging
import datetime
import numpy as np
import pandas as pd
from influxdb import DataFrameClient
from azure.datalake.store import lib
from azure.datalake.store import core
from gordo_components.dataset.base import GordoBaseDataset

logger = logging.getLogger(__name__)


def resample(tag_frame: pd.DataFrame, resolution: str) -> pd.DataFrame:
    """
    Resample dataframe with a given resolution, using mean as the aggregating function.

    Parameters
    ----------
    tag_frame: pd.DataFrame - dataframe to be resampled.
    resolution: str - using Pandas resample nomenclature (e.g. "10T" is 10 minutes).

    Returns
    -------
    pd.DataFrame: Resampled dataframe. Contains NaN if time groups are empty.

    """
    tag_frame = tag_frame.resample(resolution).mean()
    return tag_frame


def fillnan(tag_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Forward fill NaN in (typically resampled) dataframe.

    Parameters
    ----------
    tag_frame: pd.DataFrame - dataframe to be filled.

    Returns
    -------
    pd.DataFrame: Frame where the NaNs have been filled
    """
    tag_frame = tag_frame.fillna(method="ffill")
    return tag_frame


def join_resampled_frames(tag_frames: list) -> pd.DataFrame:
    """
    Join a list of Pandas dataframes with another.
    The start and end timestamps of the dataframes might differ.
    We select only the common range immediately by using an "inner" join.

    Parameters
    ----------
    tag_frames: list of pd.DataFrame - list of (resampled)
                dataframes (for different tags)

    Returns
    -------
    pd.DataFrame: One dataframe with as many columns as elements in the original list

    """
    return pd.concat(tag_frames, axis=1, join="inner")


def get_datalake_token(interactive=True, dl_service_auth_str=None):
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


class DataLakeBackedDataset(GordoBaseDataset):
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
        datalake_config: dict,
        from_ts: datetime.datetime,
        to_ts: datetime.datetime,
        tag_list: list,
        resolution: str = "10T",
    ):
        """
        Instantiates a DataLakeBackedDataset, for fetching of data from the data lake
        Parameters
        ----------
        datalake_config: dict - Datalake specific information, in particular
                        'dl_service_auth_str' and 'storename'
                        The 'dl_service_auth_str' is a string on the format
                        'tenant_id:service_id:service_secret'.
                        These parameters must be obtained from the DataLake admins,
                        or a key-store or similar.
        from_ts: datetime.datetime - start time for returned data (inclusive)
        to_ts: datetime.datetime - end time for returned data (non-inclusive)
        tag_list: list - list of tags to fetch data for
        resolution: str - the resolution to be used for resampling, in Pandas syntax
        """
        self.datalake_config = datalake_config
        self.from_ts = from_ts
        self.to_ts = to_ts
        self.tag_list = tag_list
        self.resolution = resolution
        self.interactive = self.datalake_config.get("interactive", False)

        if not self.from_ts.tzinfo or not self.to_ts.tzinfo:
            raise ValueError(
                f"Timestamps ({self.from_ts}, {self.to_ts}) need to include timezone "
                f"information"
            )

    def get_data(self) -> pd.DataFrame:
        """
        Based on the config set in the constructor, return a data-frame with all data.
        The data will be resampled, and not contain NaNs.
        Returns
        -------
        pd.DataFrame: The data frame with all tags.
        """
        adls_file_system_client = self.create_adls_client(self.interactive)

        years = range(self.from_ts.year, self.to_ts.year + 1)

        resampled_tag_frames = []
        for tag in self.tag_list:
            logger.info(f"Processing tag {tag}")

            tag_frame_all_years = self.read_tag_files(
                adls_file_system_client, tag, years
            )
            filtered = tag_frame_all_years[
                (tag_frame_all_years.index >= self.from_ts)
                & (tag_frame_all_years.index < self.to_ts)
            ]
            resampled = resample(filtered, self.resolution)
            filled = fillnan(resampled)
            resampled_tag_frames.append(filled)

        X = join_resampled_frames(resampled_tag_frames)
        y = None
        return X, y

    def create_adls_client(self, interactive) -> core.AzureDLFileSystem:
        """
        Create ADLS file system client based on the 'datalake_config' object

        Returns
        -------
        core.AzureDLFileSystem: Instance of AzureDLFileSystem, ready for use
        """
        azure_data_store = self.datalake_config.get("storename")

        if interactive:
            token = get_datalake_token(interactive=True)

        else:
            dl_service_auth_str = self.datalake_config.get(
                "dl_service_auth_str", os.environ.get("DL_SERVICE_AUTH_STR")
            )
            token = get_datalake_token(
                interactive=False, dl_service_auth_str=dl_service_auth_str
            )

        adls_file_system_client = core.AzureDLFileSystem(
            token, store_name=azure_data_store
        )
        return adls_file_system_client

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
                    names=["Sensor", "Value", "Timestamp", "Status"],
                    usecols=["Value", "Timestamp"],
                    dtype={"Value": np.float32},
                    parse_dates=["Timestamp"],
                    date_parser=lambda col: pd.to_datetime(col, utc=True),
                    index_col="Timestamp",
                )

                # There comes duplicated timestamps, keep the last
                if df.index.duplicated().any():
                    df = df[~df.index.duplicated(keep="last")]

                all_years.append(df)
                logger.info(f"Done parsing file {file_path}")

        combined = pd.concat(all_years)
        return combined

    def get_metadata(self):
        metadata = {
            "tag_list": self.tag_list,
            "train_start_date": self.from_ts,
            "train_end_date": self.to_ts,
            "resolution": self.resolution,
        }
        return metadata

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


class InfluxBackedDataset(GordoBaseDataset):
    def __init__(
        self,
        influx_config,
        from_ts,
        to_ts,
        tag_list=None,
        resolution="10m",
        resample=True,
        **kwargs,
    ):
        """
        TODO: Finish docs

        Parameters
        ----------
            influx_config: dict - Configuration for InfluxDB connection with keys:
                host, port, username, password, database
            tag_list: List[str] - List of tags
            from_ts: timestamp: start date of training period
            to_ts  : timestamp: end date of training period
            resolution: str - ie. "10m"
            resample: bool - Whether to resample.
        """
        self.to_ts = to_ts
        self.tag_list = tag_list
        self.from_ts = from_ts
        self.resample = resample
        self.resolution = resolution
        self.influx_config = influx_config
        self.influx_client = DataFrameClient(**influx_config)

    def get_data(self):
        X = self._get_sensor_data()
        y = None
        return X, y

    def read_single_sensor(self, tag):
        """
        Parameters
        ----------
            tag: str - The tag attached to the timeseries

        Returns
        -------
            One column DataFrame
        """

        logger.info(f"Reading tag: {tag}")
        logger.info(f"Fetching data from {self.from_ts} to {self.to_ts}")
        measurement = self.influx_config["database"]
        query_string = f"""
            SELECT {'mean("Value")' if self.resample else '"Value"'} as "{tag}" 
            FROM "{measurement}" 
            WHERE("tag" =~ /^{tag}$/) 
                {f"AND time >= {int(self.from_ts.timestamp())}s" if self.from_ts else ""} 
                {f"AND time <= {int(self.to_ts.timestamp())}s" if self.to_ts else ""} 
            {f'GROUP BY time({self.resolution}), "tag" fill(previous)' if self.resample else ""}
        """
        logger.info(f"Query string: {query_string}")
        dataframes = self.influx_client.query(query_string)

        list_of_tags = self._list_of_tags_from_influx()
        if tag not in list_of_tags:
            raise ValueError(
                f"tag {tag} is not a valid tag.  List of tags = {list_of_tags}"
            )

        try:
            vals = dataframes.values()
            result = list(vals)[0]
            return result

        except IndexError as e:
            logger.error(
                f"Unable to find data for tag {tag} in the time range {self.from_ts} - {self.to_ts}"
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

    def _get_sensor_data(self):
        """
        Returns:
            A single dataframe with all sensors. Note, due to potential leading
            NaNs (which are removed) the from_ts might differ from the
            resulting first timestamp in the dataframe
        """
        if self.tag_list is None:
            self.tag_list = []

        logger.info(f"Taglist:\n{self.tag_list}")
        sensors = []
        for tag in self.tag_list:
            single_sensor = self.read_single_sensor(tag)
            sensors.append(single_sensor)

        all_tags = pd.concat(sensors, axis=1)

        # TODO: This should be removed after mvp
        first_valid = all_tags[all_tags.notnull().all(axis=1)].index[0]
        if first_valid != all_tags.index[0]:
            logger.warning("Removing first part of data due to NaN")
            logger.warning(
                f"Starting at {first_valid} instead of " f"{all_tags.index[0]}"
            )
            all_tags = all_tags[first_valid:]

        return all_tags

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
