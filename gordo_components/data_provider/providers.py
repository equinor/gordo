# -*- coding: utf-8 -*-
import logging
import os
from datetime import datetime

import typing

from cachetools import cached, TTLCache
import pandas as pd
from influxdb import DataFrameClient

from gordo_components.data_provider.azure_utils import create_adls_client
from gordo_components.data_provider.base import GordoBaseDataProvider

from gordo_components.data_provider.iroc_reader import IrocReader
from gordo_components.data_provider.ncs_reader import NcsReader

logger = logging.getLogger(__name__)


def load_dataframes_from_multiple_providers(
    data_providers: typing.List[GordoBaseDataProvider],
    from_ts: datetime,
    to_ts: datetime,
    tag_list: typing.List[str],
) -> typing.Iterable[pd.DataFrame]:
    """
    Loads the tags in `tag_list` using multiple instances of
    :class:`gordo_components.data_provider.base.GordoBaseDataProvider` provided in the
    parameter `data_providers`. Will load a tag from the first data provider in the list
    which claims it. See
    :func:`gordo_components.data_provider.base.GordoBaseDataProvider.load_dataframes`.

    Returns
    -------
    Iterable[pd.DataFrame]
        The required tags as an iterable of dataframes where each is a single column
        dataframe with time index

    """
    readers_to_tags = {
        reader: [] for reader in data_providers
    }  # type: typing.Dict[GordoBaseDataProvider, typing.List[str]]
    for tag in tag_list:
        for tag_reader in data_providers:
            if tag_reader.can_handle_tag(tag):
                readers_to_tags[tag_reader].append(tag)
                logger.info(f"Assigning tag: {tag} to reader {tag_reader}")
                # In case of a tag matching two readers, we let the "first"
                # one handle it
                break
        # The else branch is executed if the break is not called
        else:
            raise ValueError(f"Found no data providers able to download the tag {tag}")
    for tag_reader, readers_tags in readers_to_tags.items():
        if readers_tags:
            logger.info(f"Using tag reader {tag_reader} to fetch tags {readers_tags}")
            for df in tag_reader.load_dataframes(
                from_ts=from_ts, to_ts=to_ts, tag_list=readers_tags
            ):
                yield df


class DataLakeProvider(GordoBaseDataProvider):

    _SUB_READER_CLASSES = [
        NcsReader,
        IrocReader,
    ]  # type: typing.List[typing.Type[GordoBaseDataProvider]]

    def can_handle_tag(self, tag):
        for r in self._get_sub_dataproviders():
            if r.can_handle_tag(tag):
                return True
        return False

    def __init__(
        self,
        storename: str = "dataplatformdlsprod",
        interactive: bool = False,
        dl_service_auth_str: str = None,
        **kwargs,
    ):
        """
        Instantiates a DataLakeBackedDataset, for fetching of data from the data lake

        Parameters
        ----------
        storename
            The store name to read data from
        interactive: bool
            To perform authentication interactively, or attempt to do it a
            automatically, in such a case must provide 'del_service_authS_tr'
            parameter or as 'DL_SERVICE_AUTH_STR' env var.
        dl_service_auth_str: Optional[str]
            string on the format 'tenant_id:service_id:service_secret'. To
            perform authentication automatically; will default to
            DL_SERVICE_AUTH_STR env var or None

        """
        super().__init__(**kwargs)
        self.storename = storename
        self.interactive = interactive
        self.dl_service_auth_str = dl_service_auth_str or os.environ.get(
            "DL_SERVICE_AUTH_STR"
        )
        self.client = None

    def load_dataframes(
        self, from_ts: datetime, to_ts: datetime, tag_list: typing.List[str]
    ) -> typing.Iterable[pd.DataFrame]:
        """
        See
        :func:`gordo_components.data_provider.base.GordoBaseDataProvider.load_dataframes`
        for documentation
        """
        # We create them here so we only try to get a auth-token once we actually need
        # it, otherwise we would have constructed them in the constructor.
        if to_ts < from_ts:
            raise ValueError(
                f"DataLakeReader called with to_ts: {to_ts} before from_ts: {from_ts}"
            )
        data_providers = self._get_sub_dataproviders()

        yield from load_dataframes_from_multiple_providers(
            data_providers, from_ts, to_ts, tag_list
        )

    def _get_client(self):
        if not self.client:
            self.client = create_adls_client(
                storename=self.storename,
                dl_service_auth_str=self.dl_service_auth_str,
                interactive=self.interactive,
            )
        return self.client

    def _get_sub_dataproviders(self):
        data_providers = [
            t_reader(client=self._get_client())
            for t_reader in DataLakeProvider._SUB_READER_CLASSES
        ]
        return data_providers


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
        measurement: str
            Name of the measurement to select from in Influx
        value_name: str
            Name of value to select, default to 'Value'
        api_key: str
            Api key to use in header
        api_key_header: str
            Key of header to insert the api key for requests
        kwargs: dict
            These are passed directly to the init args of influxdb.DataFrameClient
        """
        super().__init__(**kwargs)
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
        self, from_ts: datetime, to_ts: datetime, tag_list: typing.List[str]
    ) -> typing.Iterable[pd.DataFrame]:
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
            from_ts: datetime
                Datetime to start querying for data
            to_ts: datetime
                Datetime to stop query for data
            tag: str
                Name of the tag to match in influx
            measurement: str
                name of the measurement to select from
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

    @cached(cache=TTLCache(maxsize=10, ttl=600))
    def get_list_of_tags(self) -> typing.List[str]:
        """
        Queries Influx for the list of tags, using a TTL cache of 600 seconds. The
        cache can be cleared with :func:`cache_clear()` as is usual with cachetools.

        Returns
        -------
        List[str]
            The list of tags in Influx

        """
        return self._list_of_tags_from_influx()

    def can_handle_tag(self, tag):
        return tag in self.get_list_of_tags()
