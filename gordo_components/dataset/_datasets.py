# -*- coding: utf-8 -*-

import logging
import numpy as np
import pandas as pd

from datetime import timedelta
from influxdb import DataFrameClient
from gordo_components.dataset.base import GordoBaseDataset

logger = logging.getLogger(__name__)


class InfluxBackedDataset(GordoBaseDataset):


    def __init__(self,
                 influx_config,
                 from_ts,
                 to_ts,
                 tag_list=None,
                 resolution="10m",
                 resample=True,
                 **kwargs):
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

        logger.info("Reading tag: {}".format(tag))
        logger.info("Fetching data from {} to {}".format(self.from_ts, self.to_ts))
        measurement = self.influx_config["database"]
        query_string = f'''
            SELECT {'mean("Value")' if self.resample else '"Value"'} as "{tag}" 
            FROM "{measurement}" 
            WHERE("tag" =~ /^{tag}$/) 
                {f"AND time >= {int(self.from_ts.timestamp())}s" if self.from_ts else ""} 
                {f"AND time <= {int(self.to_ts.timestamp())}s" if self.to_ts else ""} 
            {f'GROUP BY time({self.resolution}), "tag" fill(previous)' if self.resample else ""}
        '''
        logger.info("Query string: {}".format(query_string))
        dataframes = self.influx_client.query(query_string)

        list_of_tags = self._list_of_tags_from_influx()
        if tag not in list_of_tags:
            raise ValueError (f'tag {tag} is not a valid tag.  List of tags = {list_of_tags}')

        try:
            vals = dataframes.values()
            result = list(vals)[0]
            return result

        except IndexError as e:
            logger.error(f"Unable to find data for tag {tag} in the time range {self.from_ts} - {self.to_ts}")
            raise e

    def _list_of_tags_from_influx(self):
        query_tags = f'''SHOW TAG VALUES ON {self.influx_config["database"]} WITH KEY="tag" '''
        result = self.influx_client.query(query_tags)
        list_of_tags = []
        for item in list(result.get_points()):
            list_of_tags.append(item['value'])
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

        logger.info("Taglist:\n{}".format(self.tag_list))
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
        metadata = {'tag_list': self.tag_list,
                    'train_start_date': self.from_ts,
                    'train_end_date': self.to_ts}
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
        X = np.random.random(
            size=self.size * self.n_features).reshape(-1, self.n_features)
        y = np.random.random(size=self.size).astype(int)
        return X, X.copy()

    def get_metadata(self):
        metadata = {'size': self.size,
                    'n_features': self.n_features}
        return metadata
