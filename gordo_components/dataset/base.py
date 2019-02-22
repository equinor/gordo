# -*- coding: utf-8 -*-

import abc
import logging
from typing import Iterable
from datetime import datetime

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class GordoBaseDataset:
    @abc.abstractmethod
    def get_data(self):
        """
        Using initialized params, returns X, y as numpy arrays.

        """

    @abc.abstractmethod
    def get_metadata(self):
        """
        Return metadata about the dataset in primitive / json encode-able dict form.
        """

    @staticmethod
    def join_timeseries(
        dataframes: Iterable[pd.DataFrame],
        resampling_startpoint: datetime,
        resolution: str,
    ) -> pd.DataFrame:
        """

        Parameters
        ----------
        dataframes - An iterator supplying [timestamp, value] dataframes

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

        for dataframe in dataframes:
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

        joined = pd.concat(resampled_frames, axis=1, join="inner")
        # Before returning, delete all rows with NaN, they were introduced by the
        # insertion of NaNs in the beginning of all timeseries

        return joined.dropna()
