# -*- coding: utf-8 -*-

import unittest

import numpy as np
import pandas as pd
import dateutil.parser

from gordo_components.dataset import get_dataset
from gordo_components.dataset.base import GordoBaseDataset
from gordo_components.dataset._datasets import RandomDataset
from gordo_components.dataset._datasets import join_timeseries


class DatasetTestCase(unittest.TestCase):
    def test_random_dataset_attrs(self):
        """
        Test expected attributes
        """
        config = {"type": "RandomDataset"}

        dataset = get_dataset(config)

        self.assertTrue(isinstance(dataset, GordoBaseDataset))
        self.assertTrue(isinstance(dataset, RandomDataset))
        self.assertTrue(hasattr(dataset, "get_data"))
        self.assertTrue(hasattr(dataset, "get_metadata"))

        X, y = dataset.get_data()
        self.assertTrue(isinstance(X, np.ndarray))

        # y can either be None or an numpy array
        self.assertTrue(isinstance(y, np.ndarray) or y is None)

        metadata = dataset.get_metadata()
        self.assertTrue(isinstance(metadata, dict))

    def test_join_timeseries(self):
        timeseries_list, latest_start, earliest_end = self.create_timeseries_list()

        self.assertTrue(
            len(timeseries_list[0]) > len(timeseries_list[1]) > len(timeseries_list[2])
        )

        frequency = "7T"
        timedelta = pd.Timedelta("7 minutes")
        resampling_start = dateutil.parser.isoparse("2017-12-25 06:00:00Z")
        all_in_frame = join_timeseries(timeseries_list, resampling_start, frequency)

        # Check that first resulting resampled, joined row is within "frequency" from
        # the real first data point
        self.assertGreaterEqual(
            all_in_frame.index[0], pd.Timestamp(latest_start) - timedelta
        )
        self.assertLessEqual(all_in_frame.index[-1], pd.Timestamp(earliest_end))

    def test_join_timeseries_nonutcstart(self):
        timeseries_list, latest_start, earliest_end = self.create_timeseries_list()
        frequency = "7T"
        resampling_start = dateutil.parser.isoparse("2017-12-25 06:00:00+07:00")
        all_in_frame = join_timeseries(timeseries_list, resampling_start, frequency)
        self.assertEqual(len(all_in_frame), 413)

    def test_join_timeseries_with_gaps(self):
        timeseries_list, latest_start, earliest_end = self.create_timeseries_list()

        self.assertTrue(
            len(timeseries_list[0]) > len(timeseries_list[1]) > len(timeseries_list[2])
        )

        timeseries_with_holes = self.remove_data(
            timeseries_list,
            remove_from="2018-01-03 10:00:00Z",
            remove_to="2018-01-03 18:00:00Z",
        )

        frequency = "10T"
        resampling_start = dateutil.parser.isoparse("2017-12-25 06:00:00Z")
        all_in_frame = join_timeseries(
            timeseries_with_holes, resampling_start, frequency
        )
        self.assertEqual(all_in_frame.index[0], pd.Timestamp(latest_start))
        self.assertEqual(all_in_frame.index[-1], pd.Timestamp(earliest_end))

        expected_index = pd.date_range(
            start=latest_start, end=earliest_end, freq=frequency
        )
        self.assertListEqual(list(all_in_frame.index), list(expected_index))

    @staticmethod
    def remove_data(timeseries_list, remove_from, remove_to):
        timeseries_with_holes = []
        for timeseries in timeseries_list:
            timeseries = timeseries[
                (timeseries.index < remove_from) | (timeseries.index >= remove_to)
            ]
            timeseries_with_holes.append(timeseries)

        return timeseries_with_holes

    @staticmethod
    def create_timeseries_list():
        # Create three dataframes with different resolution and different start/ends
        # Test for no NaNs, test for correct first and last date
        latest_start = "2018-01-03 06:00:00Z"
        earliest_end = "2018-01-05 06:00:00Z"

        index_seconds = pd.date_range(
            start="2018-01-01 06:00:00Z", end="2018-01-07 06:00:00Z", freq="S"
        )
        index_minutes = pd.date_range(
            start="2017-12-28 06:00:00Z", end=earliest_end, freq="T"
        )
        index_hours = pd.date_range(
            start=latest_start, end="2018-01-12 06:00:00Z", freq="H"
        )

        timeseries_seconds = pd.DataFrame(
            data=np.random.randint(0, 100, len(index_seconds)), index=index_seconds
        )
        timeseries_minutes = pd.DataFrame(
            data=np.random.randint(0, 100, len(index_minutes)), index=index_minutes
        )
        timeseries_hours = pd.DataFrame(
            data=np.random.randint(0, 100, len(index_hours)), index=index_hours
        )

        return (
            [timeseries_seconds, timeseries_minutes, timeseries_hours],
            latest_start,
            earliest_end,
        )
