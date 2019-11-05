# -*- coding: utf-8 -*-

import io
import pprint
from typing import Tuple, Union, Optional, Dict
from collections import namedtuple

from influxdb import DataFrameClient, InfluxDBClient


class EndpointMetadata:
    def __init__(self, data: dict):
        """
        Represents a traverseable dict which has come from ML server's metadata

        Examples
        --------
        >>> data = {
        ...     "key1": 1,
        ...     "key2": [1, 2, 3],
        ...     "key3": {"subkey3": "value"},
        ...     "key4": {"contains-hyphen": True}
        ... }
        >>> epm = EndpointMetadata(data)
        >>> assert epm.key1 == 1
        >>> assert len(epm.key2) == 3
        >>> assert epm.key3.subkey3 == "value"
        >>> assert epm.key4.contains_hyphen == True
        >>> print(epm)
        EndpointMetadata(data={'key1': 1,
         'key2': [1, 2, 3],
         'key3': {'subkey3': 'value'},
         'key4': {'contains-hyphen': True}}
        )
        >>> # Access the raw data under a key by calling it.
        >>> epm.key4()
        {'contains-hyphen': True}
        """
        self.__data = data

    def __getattr__(self, key):

        # If this item is has an underscore, to keep it valid python identifier,
        # we'll look for the hyphenated version in the data.
        if key.replace("_", "-") in self.__data:
            item = self.__data[key.replace("_", "-")]
        elif key in self.__data:
            item = self.__data[key]

        # Attempting to access a key which doesn't exist should act like .get(), return None
        else:
            return None

        # Depending on the type, determines if we return a new instance of this class
        if not any(isinstance(item, Obj) for Obj in (list, dict)):
            return item
        else:
            return EndpointMetadata(item)

    def __call__(self):
        """
        Calling on the instance should just return the raw data
        """
        return self.__data

    def __eq__(self, other):
        return self.__data == other.__data

    def __len__(self):
        return len(self.__data)

    def __getitem__(self, key):
        item = self.__data[key]
        if not any(isinstance(item, Obj) for Obj in (list, dict)):
            return item
        else:
            return EndpointMetadata(item)

    def __repr__(self):
        buff = io.StringIO()
        pprint.pprint(self.__data, stream=buff)
        buff.seek(0)
        return f"EndpointMetadata(data={buff.read()})"


# Prediction result representation, name=str, predictions=dataframe, error_messages=List[str]
PredictionResult = namedtuple("PredictionResult", "name predictions error_messages")


def _parse_influx_uri(uri: str) -> Tuple[str, str, str, str, str, str]:
    """
    Parse an influx URI

    Parameters
    ----------
    uri: str
        Format: <username>:<password>@<host>:<port>/<optional-path>/<db_name>

    Returns
    -------
    (str, str, str, str, str, str)
        username, password, host, port, path, database
    """
    username, password, host, port, *path, db_name = (
        uri.replace("/", ":").replace("@", ":").split(":")
    )
    path_str = "/".join(path) if path else ""
    return username, password, host, port, path_str, db_name


def influx_client_from_uri(
    uri: str,
    api_key: Optional[str] = None,
    api_key_header: Optional[str] = "Ocp-Apim-Subscription-Key",
    recreate: bool = False,
    dataframe_client: bool = False,
    proxies: Dict[str, str] = {"https": "", "http": ""},
) -> Union[InfluxDBClient, DataFrameClient]:
    """
    Get a InfluxDBClient or DataFrameClient from a SqlAlchemy like URI

    Parameters
    ----------
    uri: str
        Connection string format: <username>:<password>@<host>:<port>/<optional-path>/<db_name>
    api_key: str
        Any api key required for the client connection
    api_key_header: str
        The name of the header the api key should be assigned
    recreate: bool
        Re/create the database named in the URI
    dataframe_client: bool
        Return a DataFrameClient instead of a standard InfluxDBClient
    proxies: dict
        A mapping of any proxies to pass to the influx client

    Returns
    -------
    Union[InfluxDBClient, DataFrameClient]
    """

    username, password, host, port, path, db_name = _parse_influx_uri(uri)

    Client = DataFrameClient if dataframe_client else InfluxDBClient

    client = Client(
        host=host,
        port=port,
        database=db_name,
        username=username,
        password=password,
        path=path,
        ssl=bool(api_key),
        proxies=proxies,
    )
    if api_key:
        client._headers[api_key_header] = api_key
    if recreate:
        client.drop_database(db_name)
        client.create_database(db_name)
    return client
