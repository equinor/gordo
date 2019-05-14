# -*- coding: utf-8 -*-

import asyncio
import logging
import typing
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from typing_extensions import Protocol

from gordo_components.client.utils import influx_client_from_uri, EndpointMetadata


"""
Module contains objects which can be made into async generators which take
and EndpointMetadata and metadata (dict) when instantiated and are sent
prediction dataframes as the prediction client runs::

    async def my_forwarder(
        predictions: pd.DataFrame = None,
        endpoint: EndpointMetadata = None,
        metadata: dict = dict(),
        resampled_sensor_data: pd.DataFrame = None
    ):
        ...

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
        being sent dataframes generated from the '/anomaly/prediction' endpoint

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
        # Create df client if provided
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
        Takes a multi-layed column dataframe and write points to influx where
        each top level name is treated as the measurement name.

        Parameters
        ----------
        predictions: pd.DataFrame
            Multi layed column dataframe, where top level names will be treated
            as the 'measurement' name in influx and 2nd level will be the fields
            under those measurements.

        Returns
        -------
        None
        """
        # Setup tags; metadata (if any) and other key value pairs.
        tags = {"machine": f"{endpoint.target_name}"}
        tags.update(metadata)

        # The measurements to be posted to Influx
        top_lvl_names = predictions.columns.get_level_values(0).unique()

        for top_lvl_name in top_lvl_names:

            # Makes no sense to create a influx point where the field would be a date
            # and doesn't seem to be possible anyway.
            if top_lvl_name in ["end", "start"]:
                continue

            # this is a 'regular' non-stacked column dataframe
            sub_df = predictions[top_lvl_name]

            # Set the sub df's column names equal to the name of the tags if
            # they match the length of the tag list.
            if len(sub_df.columns) == len(endpoint.tag_list):
                sub_df.columns = [tag.name for tag in endpoint.tag_list]

            # Async write predictions to influx
            with ThreadPoolExecutor(max_workers=1) as executor:
                logger.info(
                    f"Writing {len(sub_df)} points to Influx for measurement: {top_lvl_name}"
                )
                future = executor.submit(
                    self.dataframe_client.write_points,
                    dataframe=sub_df,
                    measurement=top_lvl_name,
                    tags=tags,
                    batch_size=10000,
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
