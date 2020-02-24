# -*- coding: utf-8 -*-

import abc
import itertools
import logging
import time
from typing import Any, Dict, Optional

import pandas as pd
import numpy as np

from gordo.client.utils import influx_client_from_uri
from gordo.machine import Machine


logger = logging.getLogger(__name__)


class PredictionForwarder(metaclass=abc.ABCMeta):

    """
    Definition of a callable which the :class:`gordo.client.Client`
    will call after each successful prediction response::

        def my_forwarder(
            predictions: pd.DataFrame = None,
            machine: Machine = None,
            metadata: dict = dict(),
            resampled_sensor_data: pd.DataFrame = None
        ):
            ...
    """

    @abc.abstractmethod
    def __call__(
        self,
        *,
        predictions: pd.DataFrame = None,
        machine: Machine = None,
        metadata: dict = dict(),
        resampled_sensor_data: pd.DataFrame = None,
    ):
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
        n_retries=5,
    ):
        """
        Create an instance which, when called, is a coroutine capable of
        being sent dataframes generated from the '/anomaly/prediction' machine

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
        self.n_retries = n_retries
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

    def __call__(
        self,
        *,
        predictions: pd.DataFrame = None,
        machine: Machine = None,
        metadata: dict = dict(),
        resampled_sensor_data: pd.DataFrame = None,
    ):
        # clean predictions for possible inf, nan, which influx can't handle
        if predictions is not None:
            predictions = self._clean_df(predictions)
        if resampled_sensor_data is not None:
            resampled_sensor_data = self._clean_df(resampled_sensor_data)

        if resampled_sensor_data is None and predictions is None:
            raise ValueError(
                "Argument `resampled_sensor_data` or `predictions` must be passed"
            )
        if predictions is not None:
            if machine is None:
                raise ValueError(
                    "Argument `machine`must be provided if `predictions` is provided"
                )
            self.forward_predictions(predictions, machine=machine, metadata=metadata)
        if resampled_sensor_data is not None:
            self.send_sensor_data(resampled_sensor_data)

    @staticmethod
    def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure dataframe doesn't have inf / nan values which influx can't handle

        Parameters
        ----------
        df: pandas.DataFrame

        Returns
        -------
        pandas.DataFrame
        """
        return df.replace([np.inf, -np.inf], np.nan).dropna()

    def forward_predictions(
        self, predictions: pd.DataFrame, machine: Machine, metadata: dict = dict()
    ):
        """
        Takes a multi-layed column dataframe and write points to influx where
        each top level name is treated as the measurement name.

        How the data is written via InfluxDB DataFrameClient in this method
        determines the schema of the database.

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
        tags = {"machine": f"{machine.name}"}
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

            if isinstance(sub_df, pd.Series):
                sub_df = pd.DataFrame(sub_df)

            # Set the sub df's column names equal to the name of the tags if
            # they match the length of the tag list.
            if len(sub_df.columns) == len(machine.dataset.tag_list):
                sub_df.columns = [tag.name for tag in machine.dataset.tag_list]

            self._write_to_influx_with_retries(sub_df, top_lvl_name, tags)

    def _write_to_influx_with_retries(self, df, measurement, tags: Dict[str, Any] = {}):
        """
        Write data to influx with retries and exponential backof. Will sleep
        exponentially longer between each retry, starting at 8 seconds, capped at 5 min.
        """
        logger.info(
            f"Writing {len(df)} points to Influx for measurement: {measurement}"
        )

        for current_attempt in itertools.count(start=1):
            try:
                df = ForwardPredictionsIntoInflux._stack_to_name_value_columns(df)

                self.dataframe_client.write_points(
                    dataframe=df,
                    measurement=measurement,
                    tags=tags,
                    tag_columns=["sensor_name"],
                    field_columns=["sensor_value"],
                    batch_size=10000,
                )
            except Exception as exc:
                if current_attempt <= self.n_retries:
                    # Sleep at most 5 min
                    time_to_sleep = min(2 ** (current_attempt + 2), 300)
                    logger.warning(
                        f"Failed to forward data to influx on attempt "
                        f"{current_attempt} out of {self.n_retries}.\n"
                        f"Error: {exc}.\n"
                        f"Sleeping {time_to_sleep} seconds and trying again."
                    )
                    time.sleep(time_to_sleep)
                    continue
                else:
                    msg = f"Failed to forward data to influx. Error: {exc}"
                    logger.error(msg)
            else:
                break

    def send_sensor_data(self, sensors: pd.DataFrame):
        """
        Write sensor-data to influx
        """
        # Write the per-sensor points to influx
        logger.info(f"Writing {len(sensors)} sensor points to Influx")
        self._write_to_influx_with_retries(sensors, "resampled")
        logger.debug("Done writing resampled sensor values to influx")

    @staticmethod
    def _stack_to_name_value_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Stack a DataFrame with a range of tag columns to one with name and value columns

        Parameters
        ----------
        df: pd.DataFrame
            Source dataframe with individual columns for tags.

        Returns
        -------
        df: pd.DataFrame
            Stacked dataframe with columns `sensor_name` and `sensor_value`.
        """
        # String column names are necessary for stacking
        # (as opposed to integers when df created from np.ndarray)
        df.columns = df.columns.astype(str)

        df = df.stack().to_frame(name="sensor_value")
        df = df.reset_index(level=1).rename(columns={"level_1": "sensor_name"})
        df["sensor_value"].astype(float)
        return df
