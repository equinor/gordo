# -*- coding: utf-8 -*-

import io
import pprint
from typing import Tuple, Union, Optional, Dict, List
from collections import namedtuple

from influxdb import DataFrameClient, InfluxDBClient

from gordo_components.dataset.sensor_tag import normalize_sensor_tags, SensorTag


class EndpointMetadata:
    def __init__(self, data: dict):
        """
        Keeps easy access to common endpoint data attributes, the raw data
        being accessible via ``EndpointMetadata.data()``
        """
        self.__data = data

    def raw_metadata(self):
        """
        Access the raw metadata
        """
        return self.__data.copy()

    @property
    def name(self):
        """Name of this endpoint"""
        return self.__data["endpoint-metadata"]["metadata"]["name"]

    @property
    def endpoint(self):
        """
        The path to the endpoint, *not* including the base url
        ie. /gordo/v0/project-name/target-name
        """
        return self.__data["endpoint"]

    @property
    def tag_list(self) -> List[SensorTag]:
        """
        List of the input tags for the model
        """
        return normalize_sensor_tags(
            self.__data["endpoint-metadata"]["metadata"]["dataset"]["tag_list"]
        )

    @property
    def target_tag_list(self) -> List[SensorTag]:
        """
        List of the target tags for the model
        """
        return normalize_sensor_tags(
            self.__data["endpoint-metadata"]["metadata"]["dataset"]["target_tag_list"]
        )

    @property
    def resolution(self):
        """
        Resolution used in aggregation of the data
        """
        return self.__data["endpoint-metadata"]["metadata"]["dataset"]["resolution"]

    @property
    def model_offset(self):
        """
        Any model offset to be expected when getting predictions.
        """
        return self.__data["endpoint-metadata"]["metadata"]["model"].get(
            "model-offset", 0
        )

    @property
    def healthy(self):
        """
        Whether this endpoint is considered available to accept requests.
        """
        return self.__data["healthy"]

    def __eq__(self, other):
        return self.__data == other.__data

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
