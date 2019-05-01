# -*- coding: utf-8 -*-

import asyncio
import logging
import typing
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from typing_extensions import Protocol

from gordo_components.client.utils import influx_client_from_uri, EndpointMetadata


"""
Module contains objects which can be made into async generators which take
and EndpointMetadata and metadata (dict) when instantiated and are sent
prediction dataframes as the prediction client runs::

    async def my_forwarder(endpoint: Endpoint, df: pd.DataFrame):
        ...
        
    forwarder = my_forwarder(endpoint, metadata)
    forwarder.asend(None)  # Start the generator
    forwarder.asend(df)

The gordo_components.client.utils.EndpointMetadata hold information about 
what endpoint the coroutine should concern itself with, and metadata are 
keyvalue pairs
"""


logger = logging.getLogger(__name__)


class PredictionForwarder(Protocol):
    def __call__(
        self,
        *,
        predictions: pd.DataFrame = None,
        endpoint: EndpointMetadata = None,
        metadata: dict = dict(),
        resampled_sensor_data: pd.DataFrame = None,
    ) -> typing.Awaitable[None]:
        ...


class ForwardPredictionsIntoInflux(PredictionForwarder):
    """
    To be used as a 'forwarder' for the prediction client

    After instantiation, it is a coroutine which accepts prediction dataframes
    which it will pass onto influx
    """

    def __init__(
        self,
        destination_influx_uri: Optional[str] = None,
        destination_influx_api_key: Optional[str] = None,
        destination_influx_recreate: bool = False,
    ):
        """
        Create an instance which, when called, is a coroutine capable of
        being sent autoencoder prediction dataframes in which it will forward to influx

        By autoencoder prediction dataframes, we mean the columns are prefixed with 'output_'
        an 'input_' and then the tag/sensor name; and has a DatetimeIndex

        Parameters
        ----------
        destination_influx_uri: str
            Connection string for destination influx -
            format: <username>:<password>@<host>:<port>/<optional-path>/<db_name>
        destination_influx_api_key: str
            API key if needed for destination db
        destination_influx_recreate: bool
            Drop the database before filling it with data?
        """
        # Create clients if provided
        self.destionation_client = (
            influx_client_from_uri(
                destination_influx_uri,
                api_key=destination_influx_api_key,
                recreate=destination_influx_recreate,
            )
            if destination_influx_uri
            else None
        )
        self.dataframe_client = (
            influx_client_from_uri(
                destination_influx_uri,
                api_key=destination_influx_api_key,
                recreate=destination_influx_recreate,
                dataframe_client=True,
            )
            if destination_influx_uri
            else None
        )

    @staticmethod
    def create_anomaly_point_per_sensor(
        record: pd.Series, metadata: dict, endpoint: EndpointMetadata
    ):
        """
        Create JSON prepared anomaly points per sensor
        Specicially for a pandas dataframe where we have access to
        both input and output sensor values.

        Column labels are expected to be 'input_<sensor-name>', 'output_<sensor-name>', etc.

        Parameters
        ----------
        record: pd.Series
            A single row in dict form from pandas dataframe

        Returns
        -------
        List[dict]
            List of points ready to be written to influx
        """
        results = []
        # Dealing with input_/output_ predictions dataframe
        if any(c.startswith("input_") for c in record.keys() if isinstance(c, str)):
            for sensor in endpoint.tag_list:
                input = record[f"input_{sensor.name}"]
                output = record[f"output_{sensor.name}"]
                error = abs(input - output)

                # Setup tags; metadata (if any) and other key value pairs.
                tags = {
                    "machine": f"{endpoint.target_name}",
                    "sensor": f"{sensor.name}",
                    "prediction-method": "POST",
                }
                tags.update({k: v for k, v in metadata})

                # The actual record to be posted to Influx
                data_per_machine_tag = {
                    "measurement": "predictions",
                    "tags": tags,
                    "fields": {"input": input, "output": output, "error": error},
                    "time": f"{record.name}",
                }
                results.append(data_per_machine_tag)

        # Dealing with a simple sensor-name-as-column-name predictions dataframe
        else:
            # Setup tags; metadata (if any) and other key value pairs.
            tags = {"machine": f"{endpoint.target_name}", "prediction-method": "GET"}
            tags.update({k: v for k, v in metadata})

            # The actual record to be posted to Influx
            data_per_machine_tag = {
                "measurement": "predictions",
                "tags": tags,
                "fields": {
                    sensor.name: record[sensor.name] for sensor in endpoint.tag_list
                },
                "time": f"{record.name}",
            }
            results.append(data_per_machine_tag)

        return results

    @staticmethod
    def create_anomaly_point(
        record: pd.Series, metadata: dict, endpoint: EndpointMetadata
    ):
        """
        Create a single point for influx entry with this record.

        Provides a measurement 'anomaly' with field 'error' for the current
        target/machine name.

        Parameters
        ----------
        record: pd.Series
            One row from a panadas dataframe in dict form

        Returns
        -------
        dict
            One point ready to be written to influx
        """
        # Setup tags; metadata (if any) and other key value pairs.
        tags = {"machine": f"{endpoint.target_name}"}
        tags.update({k: v for k, v in metadata})

        # Dealing with input_/output_ predictions dataframe
        if any(c.startswith("input_") for c in record.keys() if isinstance(c, str)):

            tags.update({"prediction-method": "POST"})

            inputs = np.array(
                [record[k] for k in record.keys() if k.startswith("input_")]
            )
            outputs = np.array(
                [record[k] for k in record.keys() if k.startswith("output_")]
            )

            anomaly_point = {
                "measurement": "anomaly",
                "tags": tags,
                "fields": {"error": np.linalg.norm(inputs - outputs)},
                "time": f"{record.name}",
            }

        # Dealing with a simple sensor-name-as-column-name predictions dataframe
        else:
            tags.update({"prediction-method": "GET"})

            anomaly_point = {
                "measurement": "anomaly",
                "tags": tags,
                "fields": {"total_anomaly": record["total_anomaly"]},
                "time": f"{record.name}",
            }

        return anomaly_point

    async def __call__(
        self,
        *,
        predictions: pd.DataFrame = None,
        endpoint: EndpointMetadata = None,
        metadata: dict = dict(),
        resampled_sensor_data: pd.DataFrame = None,
    ):
        if resampled_sensor_data is None and predictions is None:
            raise ValueError(
                "Argument `resampled_sensor_data` or `predictions` must be passed"
            )
        if predictions is not None:
            if endpoint is None:
                raise ValueError(
                    "Argument `endpoint`must be provided if `predictions` is provided"
                )
            await self.forward_predictions(
                predictions, endpoint=endpoint, metadata=metadata
            )
        if resampled_sensor_data is not None:
            await self.send_sensor_data(resampled_sensor_data)

    async def forward_predictions(
        self,
        predictions: pd.DataFrame,
        endpoint: EndpointMetadata,
        metadata: dict = dict(),
    ):
        """
        Async write predictions to influx
        """
        # First, let's post all anomalies per sensor
        logger.info(f"Calculating points per sensor per record")
        data = [
            point
            for package in predictions.apply(
                lambda rec: self.create_anomaly_point_per_sensor(
                    rec, metadata, endpoint
                ),
                axis=1,
            )
            for point in package
        ]
        # Async write predictions to influx
        with ThreadPoolExecutor(max_workers=1) as executor:
            # Write the per-sensor points to influx
            logger.info(f"Writing {len(data)} sensor points to Influx")
            future = executor.submit(
                self.destionation_client.write_points, data, batch_size=10000
            )
            await asyncio.wrap_future(future)

            # Now calculate the error per line from model input vs output
            logger.debug(f"Calculating points per record")
            data = [
                point
                for point in predictions.apply(
                    lambda rec: self.create_anomaly_point(rec, metadata, endpoint),
                    axis=1,
                )
            ]
            logger.info(f"Writing {len(data)} points to Influx")

            # Write the per-sample errors to influx
            future = executor.submit(
                self.destionation_client.write_points, data, batch_size=10000
            )
            await asyncio.wrap_future(future)

    async def send_sensor_data(self, sensors: pd.DataFrame):
        """
        Async write sensor-data to influx
        """
        with ThreadPoolExecutor(max_workers=1) as executor:
            # Write the per-sensor points to influx
            logger.info(f"Writing {len(sensors)} sensor points to Influx")

            for tag_name, tag_data in _explode_df(sensors).items():
                future = executor.submit(
                    self.dataframe_client.write_points,
                    tag_data,
                    measurement="resampled",
                    batch_size=10000,
                    tags={"tag": tag_name},
                )
                await asyncio.wrap_future(future)
                logger.debug(f"Wrote resampled tag data for {tag_name} to Influx")
            logger.debug("Done writing resamples sensor values to influx")


def _explode_df(
    df: pd.DataFrame, field_name: str = "value"
) -> typing.Dict[str, pd.DataFrame]:
    """
    Converts a single dataframe with several columns to a map from the column names
    to dataframes which has that single column as the only column with the name
    `field_name`

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame with one or more columns
    field_name:
        Name to give the data-columns in the resulting map

    Returns
    -------
    Dict: A map from column names to DataFrames containing only that column.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"Column 1": [1,2,3], "Col2": [5,6,7]})
    >>> res = _explode_df(df)
    >>> len(res)
    2
    >>> sorted(list(res.keys()))
    ['Col2', 'Column 1']
    >>> res["Col2"]
       value
    0      5
    1      6
    2      7
    """
    ret = dict()
    for col in df.columns:
        ret[col] = df[[col]].rename(columns={col: field_name}, inplace=False)
    return ret
