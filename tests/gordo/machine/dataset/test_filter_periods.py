# -*- coding: utf-8 -*-

import pytest
from gordo.machine.dataset.datasets import RandomDataset
from gordo.machine.dataset.filter_periods import filter_periods
from gordo.machine.dataset.sensor_tag import SensorTag


@pytest.fixture
def dataset():
    return RandomDataset(
        train_start_date="2017-01-01 00:00:00Z",
        train_end_date="2018-01-01 00:00:00Z",
        tag_list=[SensorTag("Tag 1", None), SensorTag("Tag 2", None)],
    )


def test_filter_periods(dataset):
    data, _ = dataset.get_data()
    assert data.shape == (1873, 2)

    with pytest.raises(TypeError):
        filter_periods(data=data, granularity="10T", filter_method="abc", n_iqr=1)

    data_filtered = filter_periods(
        data=data, granularity="10T", filter_method="median", n_iqr=1
    )
    assert data_filtered.data.shape == (1634, 2)

    data_filtered = filter_periods(
        data=data, granularity="10T", filter_method="iforest", iforest_smooth=False
    )
    assert data_filtered.data.shape == (1816, 2)

    data_filtered = filter_periods(
        data=data, granularity="10T", filter_method="all", n_iqr=1, iforest_smooth=False
    )
    assert data_filtered.data.shape == (1588, 2)


def test_filter_periods_with_smoothing(dataset):
    data, _ = dataset.get_data()

    data_filtered = filter_periods(
        data=data, granularity="10T", filter_method="iforest", iforest_smooth=True
    )
    assert data.shape == (2364, 2)
    assert data_filtered.data.shape == (2011, 2)


def test_filter_periods_wiht_smoothing_all(dataset):
    data, _ = dataset.get_data()

    data_filtered = filter_periods(
        data=data, granularity="10T", filter_method="all", n_iqr=1, iforest_smooth=True
    )
    assert data.shape == (1837, 2)
    assert data_filtered.data.shape == (1305, 2)
