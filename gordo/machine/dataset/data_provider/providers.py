# -*- coding: utf-8 -*-
import random
import logging
import timeit

from datetime import datetime
import typing
from typing import Dict, Union, Any, Optional
from copy import copy

from cachetools import cached, TTLCache
import numpy as np
import pandas as pd
from influxdb import DataFrameClient

from gordo.machine.dataset.file_system import FileSystem
from gordo.machine.dataset.data_provider.base import GordoBaseDataProvider
from gordo.util import capture_args

from gordo.machine.dataset.data_provider.iroc_reader import IrocReader
from gordo.machine.dataset.data_provider.ncs_reader import NcsReader
from gordo.machine.dataset.sensor_tag import SensorTag
from gordo.machine.dataset.exceptions import ConfigException

from .storages import create_storage, DEFAULT_STORAGE_TYPE
from .assets_config import AssetsConfig
from .resource_assets_config import load_assets_config


logger = logging.getLogger(__name__)


class NoSuitableDataProviderError(ValueError):
    pass


def load_series_from_multiple_providers(
    data_providers: typing.List[GordoBaseDataProvider],
    train_start_date: datetime,
    train_end_date: datetime,
    tag_list: typing.List[SensorTag],
    dry_run: typing.Optional[bool] = False,
) -> typing.Iterable[pd.DataFrame]:
    """
    Loads the tags in `tag_list` using multiple instances of
    :class:`gordo.machine.dataset.data_provider.base.GordoBaseDataProvider` provided in the
    parameter `data_providers`. Will load a tag from the first data provider in the list
    which claims it. See
    :func:`gordo.machine.dataset.data_provider.base.GordoBaseDataProvider.load_series`.

    Returns
    -------
    typing.Iterable[pd.Series]
        The required tags as an iterable of series where each series contains
        the tag values along with a datetime index.

    """
    readers_to_tags = {
        reader: [] for reader in data_providers
    }  # type: typing.Dict[GordoBaseDataProvider, typing.List[SensorTag]]

    for tag in tag_list:
        for tag_reader in data_providers:
            if tag_reader.can_handle_tag(tag):
                readers_to_tags[tag_reader].append(tag)
                logger.debug(f"Assigning tag: {tag} to reader {tag_reader}")
                # In case of a tag matching two readers, we let the "first"
                # one handle it
                break
        # The else branch is executed if the break is not called
        else:
            raise NoSuitableDataProviderError(
                f"Found no data providers able to download the tag {tag} "
            )
    before_downloading = timeit.default_timer()
    for tag_reader, readers_tags in readers_to_tags.items():
        if readers_tags:
            logger.debug(f"Using tag reader {tag_reader} to fetch tags {readers_tags}")
            for series in tag_reader.load_series(
                train_start_date=train_start_date,
                train_end_date=train_end_date,
                tag_list=readers_tags,
                dry_run=dry_run,
            ):
                yield series
    logger.debug(
        f"Downloading all tags took {timeit.default_timer()-before_downloading} seconds"
    )


class DataLakeProvider(GordoBaseDataProvider):

    _SUB_READER_CLASSES = [
        NcsReader,
        IrocReader,
    ]  # type: typing.List[typing.Type[GordoBaseDataProvider]]

    def can_handle_tag(self, tag):
        """Implements base method, see GordoBaseDataProvider"""
        for r in self._get_sub_dataproviders():
            if r.can_handle_tag(tag):
                return True
        return False

    @capture_args
    def __init__(
        self,
        storage: Optional[Union[FileSystem, Dict[str, Any]]] = None,
        assets_config: Optional[AssetsConfig] = None,
        interactive: Optional[bool] = None,
        storename: Optional[str] = None,
        dl_service_auth_str: Optional[str] = None,
        **kwargs,
    ):
        """
        Instantiates a DataLakeBackedDataset, for fetching of data from the data lake

        Parameters
        ----------
        storage: Optional[Union[FileSystem, Dict[str, Any]]]
            DataLake config. The structure depends on which DataLake you are going to use.
        assets_config: Optional[AssetsConfig]
            Uses assets config from `gordo.machine.data_provider.resources` by default
        interactive: bool
            To perform authentication interactively, or attempt to do it a
            automatically, in such a case must provide 'del_service_authS_tr'
            parameter or as 'DL_SERVICE_AUTH_STR' env var. Only for DataLake Gen1 and will be deprecated in new versions
        storename
            The store name to read data from. Only for DataLake Gen1 and will be deprecated in new versions
        dl_service_auth_str: Optional[str]
            string on the format 'tenant_id:service_id:service_secret'. To
            perform authentication automatically; will default to
            DL_SERVICE_AUTH_STR env var or None. Only for DataLake Gen1 and will be deprecated in new versions

        .. deprecated::
            Arguments `interactive`, `storename`, `dl_service_auth_str`
        """
        if assets_config is None:
            assets_config = load_assets_config()
        self.assets_config = assets_config
        self.kwargs = kwargs

        # This arguments only preserved for back-compatibility reasons and will be removed in future versions of gordo
        self.adl1_kwargs: Dict[str, Any] = {}
        if interactive is not None:
            self.adl1_kwargs["interactive"] = interactive
        if storename is not None:
            self.adl1_kwargs["store_name"] = storename
        if dl_service_auth_str is not None:
            self.adl1_kwargs["dl_service_auth_str"] = dl_service_auth_str

        self.storage = storage
        self._storage_instance: Optional[FileSystem] = None

    def load_series(
        self,
        train_start_date: datetime,
        train_end_date: datetime,
        tag_list: typing.List[SensorTag],
        dry_run: typing.Optional[bool] = False,
    ) -> typing.Iterable[pd.Series]:
        """
        See
        :func:`gordo.machine.dataset.data_provider.base.GordoBaseDataProvider.load_series`
        for documentation
        """
        # We create them here so we only try to get a auth-token once we actually need
        # it, otherwise we would have constructed them in the constructor.
        if train_end_date < train_start_date:
            raise ValueError(
                f"DataLakeReader called with train_end_date: {train_end_date} before train_start_date: {train_start_date}"
            )
        data_providers = self._get_sub_dataproviders()

        yield from load_series_from_multiple_providers(
            data_providers, train_start_date, train_end_date, tag_list, dry_run
        )

    def _adl1_back_compatible_kwarg(
        self, storage_type: str, kwarg: Dict[str, Any]
    ) -> Dict[str, Any]:
        if storage_type == "adl1":
            if self.adl1_kwargs:
                adl1_kwarg = copy(self.adl1_kwargs)
                adl1_kwarg.update(kwarg)
                return adl1_kwarg
        else:
            if self.adl1_kwargs:
                arguments = ", ".join(self.adl1_kwargs.keys())
                raise ConfigException(
                    "%s does no support%s by storage '%s'"
                    % (arguments, "s" if len(arguments) > 1 else "", storage_type)
                )
        return kwarg

    def _instantiate_storage(
        self, storage: Optional[Union[FileSystem, Dict[str, Any]]]
    ) -> FileSystem:
        if storage is None:
            storage = {}
        if isinstance(storage, dict):
            kwargs = copy(storage)
            storage_type = kwargs.pop("type", DEFAULT_STORAGE_TYPE)
            kwargs = self._adl1_back_compatible_kwarg(storage_type, kwargs)
            return create_storage(storage_type, **kwargs)
        return storage

    def _get_storage_instance(self) -> FileSystem:
        if self._storage_instance is None:
            self._storage_instance = self._instantiate_storage(self.storage)
        return self._storage_instance

    def _get_sub_dataproviders(self):
        storage = self._get_storage_instance()
        assets_config = self.assets_config
        data_providers = []
        for t_reader in DataLakeProvider._SUB_READER_CLASSES:
            data_providers.append(
                t_reader(storage=storage, assets_config=assets_config, **self.kwargs)
            )
        return data_providers


class InfluxDataProvider(GordoBaseDataProvider):
    @capture_args
    def __init__(
        self,
        measurement: str,
        value_name: str = "Value",
        api_key: str = None,
        api_key_header: str = None,
        client: DataFrameClient = None,
        uri: str = None,
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
        uri: str
            Create a client from a URI
            format: <username>:<password>@<host>:<port>/<optional-path>/<db_name>
        kwargs: dict
            These are passed directly to the init args of influxdb.DataFrameClient
        """
        self.measurement = measurement
        self.value_name = value_name
        self.influx_client = client
        if kwargs.pop("threads", None):
            logger.warning(
                "InfluxDataProvider got parameter 'threads' which is not supported, it "
                "will be ignored."
            )

        if self.influx_client is None:
            if uri:

                # Import here to avoid any circular import error caused by
                # importing TimeSeriesDataset, which imports this provider
                # which would have imported Client via traversal of the __init__
                # which would then try to import TimeSeriesDataset again.
                from gordo.client.utils import influx_client_from_uri

                self.influx_client = influx_client_from_uri(  # type: ignore
                    uri,
                    api_key=api_key,
                    api_key_header=api_key_header,
                    dataframe_client=True,
                )
            else:
                if "type" in kwargs:
                    kwargs.pop("type")
                self.influx_client = DataFrameClient(**kwargs)
                if api_key is not None:
                    if not api_key_header:
                        raise ValueError(
                            "If supplying an api key, you must supply the header key to insert it under."
                        )
                    self.influx_client._headers[api_key_header] = api_key

    def load_series(
        self,
        train_start_date: datetime,
        train_end_date: datetime,
        tag_list: typing.List[SensorTag],
        dry_run: typing.Optional[bool] = False,
    ) -> typing.Iterable[pd.Series]:
        """
        See GordoBaseDataProvider for documentation
        """
        if dry_run:
            raise NotImplementedError(
                "Dry run for InfluxDataProvider is not implemented"
            )
        return (
            self.read_single_sensor(
                train_start_date=train_start_date,
                train_end_date=train_end_date,
                tag=tag.name,
                measurement=self.measurement,
            )
            for tag in tag_list
        )

    def read_single_sensor(
        self,
        train_start_date: datetime,
        train_end_date: datetime,
        tag: str,
        measurement: str,
    ) -> pd.Series:
        """
        Parameters
        ----------
            train_start_date: datetime
                Datetime to start querying for data
            train_end_date: datetime
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
        logger.info(f"Fetching data from {train_start_date} to {train_end_date}")
        query_string = f"""
            SELECT "{self.value_name}" as "{tag}"
            FROM "{measurement}"
            WHERE("tag" =~ /^{tag}$/)
                {f"AND time >= {int(train_start_date.timestamp())}s" if train_start_date else ""}
                {f"AND time <= {int(train_end_date.timestamp())}s" if train_end_date else ""}
        """

        logger.info(f"Query string: {query_string}")
        dataframes = self.influx_client.query(query_string)  # type: ignore

        try:
            df = list(dataframes.values())[0]
            return df[tag]

        except IndexError as e:
            list_of_tags = self._list_of_tags_from_influx()
            if tag not in list_of_tags:
                raise ValueError(f"tag {tag} is not found in influx")
            logger.error(
                f"Unable to find data for tag {tag} in the time range {train_start_date} - {train_end_date}"
            )
            raise e

    def _list_of_tags_from_influx(self):
        query_tags = (
            f"""SHOW TAG VALUES ON {self.influx_client._database} WITH KEY="tag" """
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
        typing.List[str]
            The list of tags in Influx

        """
        return self._list_of_tags_from_influx()

    def can_handle_tag(self, tag: SensorTag):
        return tag.name in self.get_list_of_tags()


class RandomDataProvider(GordoBaseDataProvider):
    """
    Get a GordoBaseDataset which returns unstructed values for X and y. Each instance
    uses the same seed, so should be a function (same input -> same output)
    """

    def can_handle_tag(self, tag: SensorTag):
        return True  # We can be random about everything

    @capture_args
    def __init__(self, min_size=100, max_size=300):
        self.max_size = max_size
        self.min_size = min_size
        np.random.seed(0)

    # Thanks stackoverflow
    # https://stackoverflow.com/questions/50559078/generating-random-dates-within-a-given-range-in-pandas
    @staticmethod
    def _random_dates(start, end, n=10):
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        start_u = start.value // 10 ** 9
        end_u = end.value // 10 ** 9

        return sorted(
            pd.to_datetime(np.random.randint(start_u, end_u, n), unit="s", utc=True)
        )

    def load_series(
        self,
        train_start_date: datetime,
        train_end_date: datetime,
        tag_list: typing.List[SensorTag],
        dry_run: typing.Optional[bool] = False,
    ) -> typing.Iterable[pd.Series]:
        if dry_run:
            raise NotImplementedError(
                "Dry run for RandomDataProvider is not implemented"
            )
        for tag in tag_list:
            nr = random.randint(self.min_size, self.max_size)

            random_index = self._random_dates(train_start_date, train_end_date, n=nr)
            series = pd.Series(
                index=random_index,
                name=tag.name,
                data=np.random.random(size=len(random_index)),
            )
            yield series
