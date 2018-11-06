# -*- coding: utf-8 -*-

import logging
import numpy as np
import pandas as pd

from collections import namedtuple
from datetime import timedelta
from influxdb import DataFrameClient
from gordo_components.dataset.base import GordoBaseDataset

logger = logging.getLogger(__name__)


class InfluxBackedDataset(GordoBaseDataset):

    Machine = namedtuple('Machine', ['machine_name', 'tag_list'])

    def __init__(self,
                 influx_config,
                 machine_name=None,
                 tag_list=None,
                 from_ts=None,
                 to_ts=None,
                 resolution="10m", 
                 resample=True,
                 **kwargs):
        """
        TODO: Finish docs

        Parameters
        ----------
            influx_config: dict - Configuration for InfluxDB connection with keys:
                host, port, username, password, database
            machine_id: str
            tag_list: List[str] - List of tags
            from_ts: Optional[timestamp]
            to_ts  : Optional[timestamp]
            resolution: str - ie. "10m"
            resample: bool - Whether to resample.
        """
        self.to_ts = to_ts
        self.machine = self.Machine(machine_name, tag_list)
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

        logger.info("Reading tag: {}".format(tag))
        logger.info("Fetching data from {} to {}".format(self.from_ts, self.to_ts))
        query_string = f'''
            SELECT {'mean("Value")' if self.resample else '"Value"'} as "{tag}" 
            FROM "{self.influx_config["database"]}" 
            WHERE("tag" =~ /^{tag}$/) 
                {f"AND time >= {int(self.from_ts.timestamp())}s" if self.from_ts else ""} 
                {f"AND time <= {int(self.to_ts.timestamp())}s" if self.to_ts else ""} 
            {f'GROUP BY time({self.resolution}), "tag" fill(previous)' if self.resample else ""}
        '''
        logger.info("Query string: {}".format(query_string))
        dataframes = self.influx_client.query(query_string)

        try:
            vals = dataframes.values()
            result = list(vals)[0]
            return result

        except (KeyError, IndexError):
            logger.error("Unable to find series with tag: {}".format(tag))

    def database_exists(self, database):
        """
        Helper: determine if a given database exists in the current influxdb connection
        """
        return any(
            entry['name'] == database for entry in self.influx_client.get_list_database()
        )

    def _get_sensor_data(self):
        """
        Args:
            machine: The machine configuration to read data for
            influx_config: Configuration of influx connection
            from_ts: Timestamp telling from which time to fetch data (optional)
            to_ts: Timestamp telling to which time to fetch data (optional)
            resolution: Desired resolution for the results eg. '10m', '10s',...
        Returns:
            A single dataframe with all sensors. Note, due to potential leading
            NaNs (which are removed) the from_ts might differ from the
            resulting first timestamp in the dataframe
        """
        logger.info("Getting data for machine {}".format(self.machine.machine_name))
        logger.info("Taglist:\n{}".format(self.machine.tag_list))
        sensors = []
        for tag in self.machine.tag_list:
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


class RandomDataset(GordoBaseDataset):
    """
    Get a GordoBaseDataset which returns random values for X and y
    """

    def __init__(self, size=100, n_features=20, **kwargs):
        self.size = size
        self.n_features = n_features

    def get_data(self):
        """return X and y data"""
        X = np.random.random(
            size=self.size * self.n_features).reshape(-1, self.n_features)
        y = np.random.random(size=self.size).astype(int)
        return X, X.copy()
