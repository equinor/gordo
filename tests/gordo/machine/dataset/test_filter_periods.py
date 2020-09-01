# -*- coding: utf-8 -*-

import pytest
from gordo.machine.dataset.datasets import RandomDataset
from gordo.machine.dataset.filter_periods import FilterPeriods
from gordo.machine.dataset.sensor_tag import SensorTag


@pytest.fixture
def dataset():
    return RandomDataset(
        train_start_date="2017-01-01 00:00:00Z",
        train_end_date="2018-01-01 00:00:00Z",
        tag_list=[SensorTag("Tag 1", None), SensorTag("Tag 2", None)],
    )


def test_filter_periods_typerror(dataset):
    data, _ = dataset.get_data()
    assert data.shape == (1873, 2)
    with pytest.raises(TypeError):
        FilterPeriods(granularity="10T", filter_method="abc", n_iqr=1)


def test_filter_periods_median(dataset):
    data, _ = dataset.get_data()
    data_filtered, drop_periods, predictions = FilterPeriods(
        granularity="10T", filter_method="median", n_iqr=1
    ).filter_data(data)

    assert data.shape == (2364, 2)
    assert tuple(data.mean().round(6)) == (0.496113, 0.516027)

    assert sum(predictions["median"]["pred"]) == -402
    assert len(drop_periods["median"]) == 35
    assert data_filtered.shape == (1962, 2)


def test_filter_periods_iforest(dataset):
    data, _ = dataset.get_data()
    data_filtered, drop_periods, predictions = FilterPeriods(
        granularity="10T", filter_method="iforest", iforest_smooth=False
    ).filter_data(data)

    assert data.shape == (1837, 2)
    assert tuple(data.mean().round(6)) == (0.519195, 0.550395)

    assert sum(predictions["iforest"]["pred"]) == 1725
    assert len(drop_periods["iforest"]) == 19
    assert data_filtered.shape == (1781, 2)


def test_filter_periods_all(dataset):
    data, _ = dataset.get_data()
    data_filtered, drop_periods, predictions = FilterPeriods(
        granularity="10T", filter_method="all", n_iqr=1, iforest_smooth=False
    ).filter_data(data)

    assert data.shape == (1516, 2)
    assert tuple(data.mean().round(6)) == (0.486544, 0.498725)

    assert sum(predictions["median"]["pred"]) == -279
    assert sum(predictions["iforest"]["pred"]) == 1424
    assert len(drop_periods["median"]) == 16
    assert len(drop_periods["iforest"]) == 11
    assert data_filtered.shape == (1219, 2)


def test_filter_periods_iforest_smoothing(dataset):
    data, _ = dataset.get_data()
    data_filtered, drop_periods, predictions = FilterPeriods(
        granularity="10T", filter_method="iforest", iforest_smooth=True
    ).filter_data(data)

    assert data.shape == (1435, 2)
    assert tuple(data.mean().round(6)) == (0.504942, 0.47524)

    assert sum(predictions["iforest"]["pred"]) == 959
    assert len(drop_periods["iforest"]) == 18
    assert data_filtered.shape == (1200, 2)


def test_filter_periods_all_smoothing(dataset):
    data, _ = dataset.get_data()
    data_filtered, drop_periods, predictions = FilterPeriods(
        granularity="10T", filter_method="all", n_iqr=1, iforest_smooth=True
    ).filter_data(data)

    assert data.shape == (1080, 2)
    assert tuple(data.mean().round(6)) == (0.496644, 0.492348)

    assert sum(predictions["iforest"]["pred"]) == 632
    assert len(drop_periods["median"]) == 15
    assert len(drop_periods["iforest"]) == 24
    assert data_filtered.shape == (767, 2)
