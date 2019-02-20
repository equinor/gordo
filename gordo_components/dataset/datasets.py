# -*- coding: utf-8 -*-

import os
import re
from typing import Iterable
import logging
import numpy as np
import pandas as pd
from influxdb import DataFrameClient
from azure.datalake.store import lib
from azure.datalake.store import core
from datetime import datetime

from gordo_components.dataset.base import GordoBaseDataset

logger = logging.getLogger(__name__)


def join_timeseries(
    dataframe_iterator: Iterable[pd.DataFrame],
    resampling_startpoint: datetime,
    resolution: str,
) -> pd.DataFrame:
    """

    Parameters
    ----------
    dataframe_iterator - An iterator supplying [timestamp, value] dataframes

    resampling_startpoint - The starting point for resampling. Most data
    frames will not have this in their datetime index, and it will be inserted with a
    NaN as the value.
    The resulting NaNs will be removed, so the only important requirement for this is
    that this resampling_startpoint datetime must be before or equal to the first
    (earliest) datetime in the data to be resampled.

    resolution - The bucket size for grouping all incoming time data (e.g. "10T")

    Returns
    -------
    pd.DataFrame - A dataframe without NaNs, a common time index, and one column per
    element in the dataframe_generator

    """
    resampled_frames = []

    for dataframe in dataframe_iterator:
        if dataframe is None:
            continue

        startpoint_sametz = resampling_startpoint.astimezone(
            tz=dataframe.index[0].tzinfo
        )
        if dataframe.index[0] > startpoint_sametz:
            # Insert a NaN at the startpoint, to make sure that all resampled
            # indexes are the same. This approach will "pad" most frames with
            # NaNs, that will be removed at the end.
            startpoint = pd.DataFrame(
                [np.NaN], index=[startpoint_sametz], columns=dataframe.columns
            )
            dataframe = startpoint.append(dataframe)
            logging.debug(
                f"Appending NaN to {dataframe.columns[0]} "
                f"at time {startpoint_sametz}"
            )

        elif dataframe.index[0] < resampling_startpoint:
            msg = (
                f"Error - for {dataframe.columns[0]}, first timestamp "
                f"{dataframe.index[0]} is before resampling start point "
                f"{startpoint_sametz}"
            )
            logging.error(msg)
            raise RuntimeError(msg)

        logging.debug("Head (3) and tail(3) of dataframe to be resampled:")
        logging.debug(dataframe.head(3))
        logging.debug(dataframe.tail(3))

        resampled = dataframe.resample(resolution).mean()

        filled = resampled.fillna(method="ffill")
        resampled_frames.append(filled)

    if len(resampled_frames) == 0:
        return None

    joined = pd.concat(resampled_frames, axis=1, join="inner")
    # Before returning, delete all rows with NaN, they were introduced by the
    # insertion of NaNs in the beginning of all timeseries

    return joined.dropna()


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
        from_ts: datetime,
        to_ts: datetime,
        tag_list: list,
        resolution: str = "10T",
        require_all_tags: bool = True,
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
        require_all_tags: bool - fail if a tag file is not found
        """
        self.datalake_config = datalake_config
        self.from_ts = from_ts
        self.to_ts = to_ts
        self.tag_list = tag_list
        self.resolution = resolution
        self.interactive = self.datalake_config.get("interactive", False)
        self.require_all_tags = require_all_tags

        if not self.from_ts.tzinfo or not self.to_ts.tzinfo:
            raise ValueError(
                f"Timestamps ({self.from_ts}, {self.to_ts}) need to include timezone "
                f"information"
            )

    def get_data(self) -> pd.DataFrame:
        dataframe_generator = self.make_dataframe_generator()

        X = join_timeseries(dataframe_generator, self.from_ts, self.resolution)
        y = None
        return X, y

    def make_dataframe_generator(self):
        """
        Based on the config set in the constructor, return a data-frame with all data.
        The data will be resampled, and not contain NaNs.
        Returns
        -------
        pd.DataFrame: The data frame with all tags.
        """
        adls_file_system_client = self.create_adls_client(self.interactive)

        years = range(self.from_ts.year, self.to_ts.year + 1)

        for tag in self.tag_list:
            logger.info(f"Processing tag {tag}")

            tag_frame_all_years = self.read_tag_files(
                adls_file_system_client, tag, years
            )
            if tag_frame_all_years is None:
                yield None

            else:
                filtered = tag_frame_all_years[
                    (tag_frame_all_years.index >= self.from_ts)
                    & (tag_frame_all_years.index < self.to_ts)
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
            if not adls_file_system_client.exists(file_path):
                if self.require_all_tags:
                    error = (
                        f"File {file_path} does not exist. Set require_all_tags to "
                        f"True to allow missing tags"
                    )
                    logger.error(error)
                    raise ValueError(error)
                else:
                    continue

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

        if len(all_years) == 0:
            return None

        combined = pd.concat(all_years)

        # There often comes duplicated timestamps, keep the last
        if combined.index.duplicated().any():
            combined = combined[~combined.index.duplicated(keep="last")]

        return combined

    def create_adls_client(self, interactive) -> core.AzureDLFileSystem:
        """
        Create ADLS file system client based on the 'datalake_config' object

        Returns
        -------
        core.AzureDLFileSystem: Instance of AzureDLFileSystem, ready for use
        """
        azure_data_store = self.datalake_config.get("storename")

        if interactive:
            token = self.get_datalake_token(interactive=True)

        else:
            dl_service_auth_str = self.datalake_config.get(
                "dl_service_auth_str", os.environ.get("DL_SERVICE_AUTH_STR")
            )
            token = self.get_datalake_token(
                interactive=False, dl_service_auth_str=dl_service_auth_str
            )

        adls_file_system_client = core.AzureDLFileSystem(
            token, store_name=azure_data_store
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
