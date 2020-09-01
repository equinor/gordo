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
        tag_list=[SensorTag("Tag 1", None)],
    )


def test_filter_periods_typerror(dataset):
    data, _ = dataset.get_data()
    assert data.shape == (9760, 1)
    with pytest.raises(TypeError):
        FilterPeriods(granularity="10T", filter_method="abc", n_iqr=1)


def test_filter_periods_median(dataset):
    data, _ = dataset.get_data()
    data_filtered, drop_periods, predictions = FilterPeriods(
        granularity="10T", filter_method="median", n_iqr=1
    ).filter_data(data)

    assert data.shape == (9063, 1)
    assert data["Tag 1"].mean() == 0.5113691034704841

    assert sum(predictions["median"]["pred"]) == -493
    assert len(drop_periods["median"]) == 44
    assert data_filtered.shape == (8570, 1)


def test_filter_periods_iforest(dataset):
    data, _ = dataset.get_data()
    data_filtered, drop_periods, predictions = FilterPeriods(
        granularity="10T", filter_method="iforest", iforest_smooth=False
    ).filter_data(data)

    assert data.shape == (12838, 1)
    assert data["Tag 1"].mean() == 0.5144733352386245

    assert sum(predictions["iforest"]["pred"]) == 12066
    assert len(drop_periods["iforest"]) == 61
    assert data_filtered.shape == (12452, 1)


def test_filter_periods_all(dataset):
    data, _ = dataset.get_data()
    data_filtered, drop_periods, predictions = FilterPeriods(
        granularity="10T", filter_method="all", n_iqr=1, iforest_smooth=False
    ).filter_data(data)

    assert data.shape == (8024, 1)
    assert data["Tag 1"].mean() == 0.500105748646813

    assert sum(predictions["median"]["pred"]) == -449
    assert sum(predictions["iforest"]["pred"]) == 7542
    assert len(drop_periods["median"]) == 39
    assert len(drop_periods["iforest"]) == 29
    assert data_filtered.shape == (7356, 1)


def test_filter_periods_iforest_smoothing(dataset):
    data, _ = dataset.get_data()
    data_filtered, drop_periods, predictions = FilterPeriods(
        granularity="10T", filter_method="iforest", iforest_smooth=True
    ).filter_data(data)

    assert data.shape == (9674, 1)
    assert data["Tag 1"].mean() == 0.5019862352609169

    assert sum(predictions["iforest"]["pred"]) == 8552
    assert len(drop_periods["iforest"]) == 41
    assert data_filtered.shape == (9113, 1)


def test_filter_periods_all_smoothing(dataset):
    data, _ = dataset.get_data()
    data_filtered, drop_periods, predictions = FilterPeriods(
        granularity="10T", filter_method="all", n_iqr=1, iforest_smooth=True
    ).filter_data(data)

    assert data.shape == (8595, 1)
    assert data["Tag 1"].mean() == 0.512856120233814

    assert sum(predictions["iforest"]["pred"]) == 7471
    assert len(drop_periods["median"]) == 39
    assert len(drop_periods["iforest"]) == 29
    assert data_filtered.shape == (7522, 1)
