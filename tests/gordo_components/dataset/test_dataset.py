# -*- coding: utf-8 -*-

import unittest
from typing import List, Iterable

import numpy as np
import pandas as pd
import dateutil.parser
from datetime import datetime

from gordo_components.data_provider.base import GordoBaseDataProvider
from gordo_components.dataset.datasets import RandomDataset, TimeSeriesDataset
from gordo_components.dataset.base import GordoBaseDataset
from gordo_components.dataset.sensor_tag import SensorTag
from gordo_components.dataset.sensor_tag import normalize_sensor_tags


class DatasetTestCase(unittest.TestCase):
    def test_random_dataset_attrs(self):
        """
        Test expected attributes
        """

        start = dateutil.parser.isoparse("2017-12-25 06:00:00Z")
        end = dateutil.parser.isoparse("2017-12-29 06:00:00Z")

        dataset = RandomDataset(
            from_ts=start,
            to_ts=end,
            tag_list=[SensorTag("Tag 1", None), SensorTag("Tag 2", None)],
        )

        self.assertTrue(isinstance(dataset, GordoBaseDataset))
        self.assertTrue(hasattr(dataset, "get_data"))
        self.assertTrue(hasattr(dataset, "get_metadata"))

        X, y = dataset.get_data()
        self.assertTrue(isinstance(X, pd.DataFrame))

        # y can either be None or an numpy array
        self.assertTrue(isinstance(y, pd.DataFrame) or y is None)

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
        resampling_end = dateutil.parser.isoparse("2018-01-15 08:00:00Z")
        all_in_frame = GordoBaseDataset.join_timeseries(
            timeseries_list, resampling_start, resampling_end, frequency
        )

        # Check that first resulting resampled, joined row is within "frequency" from
        # the real first data point
        self.assertGreaterEqual(
            all_in_frame.index[0], pd.Timestamp(latest_start) - timedelta
        )
        self.assertLessEqual(all_in_frame.index[-1], pd.Timestamp(resampling_end))

    def test_join_timeseries_nonutcstart(self):
        timeseries_list, latest_start, earliest_end = self.create_timeseries_list()
        frequency = "7T"
        resampling_start = dateutil.parser.isoparse("2017-12-25 06:00:00+07:00")
        resampling_end = dateutil.parser.isoparse("2018-01-12 13:07:00+07:00")
        all_in_frame = GordoBaseDataset.join_timeseries(
            timeseries_list, resampling_start, resampling_end, frequency
        )
        self.assertEqual(len(all_in_frame), 1854)

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
        resampling_end = dateutil.parser.isoparse("2018-01-12 07:00:00Z")

        all_in_frame = GordoBaseDataset.join_timeseries(
            timeseries_with_holes, resampling_start, resampling_end, frequency
        )
        self.assertEqual(all_in_frame.index[0], pd.Timestamp(latest_start))
        self.assertEqual(all_in_frame.index[-1], pd.Timestamp(resampling_end))

        expected_index = pd.date_range(
            start=dateutil.parser.isoparse(latest_start),
            end=resampling_end,
            freq=frequency,
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

        timeseries_seconds = pd.Series(
            data=np.random.randint(0, 100, len(index_seconds)),
            index=index_seconds,
            name="ts-seconds",
        )
        timeseries_minutes = pd.Series(
            data=np.random.randint(0, 100, len(index_minutes)),
            index=index_minutes,
            name="ts-minutes",
        )
        timeseries_hours = pd.Series(
            data=np.random.randint(0, 100, len(index_hours)),
            index=index_hours,
            name="ts-hours",
        )

        return (
            [timeseries_seconds, timeseries_minutes, timeseries_hours],
            latest_start,
            earliest_end,
        )


class TimeSeriesDatasetTest(unittest.TestCase):
    """
    Tests the TimeSeriesDataset implementation with a mock datasource
    """

    def test_row_filter(self):
        """Tests that row_filter filters away rows"""

        tag_list = [
            SensorTag("Tag 1", None),
            SensorTag("Tag 2", None),
            SensorTag("Tag 3", None),
        ]
        start = dateutil.parser.isoparse("2017-12-25 06:00:00Z")
        end = dateutil.parser.isoparse("2017-12-29 06:00:00Z")
        X, _ = TimeSeriesDataset(
            MockDataSource(), start, end, tag_list=tag_list
        ).get_data()

        self.assertEqual(577, len(X))

        X, _ = TimeSeriesDataset(
            MockDataSource(), start, end, tag_list=tag_list, row_filter="'Tag 1' < 5000"
        ).get_data()

        self.assertEqual(8, len(X))

        X, _ = TimeSeriesDataset(
            MockDataSource(),
            start,
            end,
            tag_list=tag_list,
            row_filter="'Tag 1' / 'Tag 3' < 0.999",
        ).get_data()

        self.assertEqual(3, len(X))


class MockDataSource(GordoBaseDataProvider):
    def __init__(self, **kwargs):
        pass

    def can_handle_tag(self, tag):
        return True

    def load_series(
        self, from_ts: datetime, to_ts: datetime, tag_list: List[SensorTag]
    ) -> Iterable[pd.Series]:
        days = pd.date_range(from_ts, to_ts, freq="s")
        tag_list_strings = [tag.name for tag in tag_list]
        for i, name in enumerate(tag_list_strings):
            series = pd.Series(
                index=days, data=list(range(i, len(days) + i)), name=name
            )
            yield series
