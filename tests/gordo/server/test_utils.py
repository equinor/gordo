# -*- coding: utf-8 -*-
import random
import pytest
import pandas as pd
import numpy as np
import dateutil

from gordo.server import utils as server_utils


@pytest.mark.parametrize(
    "df",
    (
        pd.DataFrame(np.random.random((10, 10))),
        pd.DataFrame(
            np.random.random((10, 10)),
            index=pd.date_range(start="2016-01-01", end="2016-01-02", periods=10),
        ),
        pd.DataFrame(
            np.random.random((10, 4)),
            columns=pd.MultiIndex.from_product((("col1", "col2"), ("ft1", "ft2"))),
        ),
    ),
)
def test_dataframe_parquet_serializers(df):
    """The (de)serialization functions should be interoperable"""
    serialized = server_utils.dataframe_into_parquet_bytes(df.copy())
    df_clone = server_utils.dataframe_from_parquet_bytes(serialized)
    assert df.columns.tolist() == df_clone.columns.tolist()
    assert df.index.tolist() == df_clone.index.tolist()
    assert np.allclose(df.values, df_clone.values)


@pytest.mark.parametrize(
    "df",
    [
        # Multi-column
        pd.DataFrame(
            np.random.random((10, 4)),
            columns=pd.MultiIndex.from_product(
                (("feature1", "feature2"), ("col1", "col2"))
            ),
            index=pd.date_range("2016-01-01", "2016-02-01", periods=10),
        ),
        # Normal dataframe, no date index
        pd.DataFrame(
            np.random.random((10, 4)), columns=["col1", "col2", "col3", "col4"]
        ),
    ],
)
def test_dataframe_from_to_dict(df):
    """
    Test (de)serializations back and forth between dataframe -> dict -> dataframe
    """
    index_was_datetimes: bool = isinstance(df.index, pd.DatetimeIndex)

    cloned = server_utils.dataframe_from_dict(server_utils.dataframe_to_dict(df))

    if index_was_datetimes:
        # Ensure the function hasn't mutated the index.
        assert isinstance(df.index, pd.DatetimeIndex)

    assert np.allclose(df.values, cloned.values)
    assert df.columns.tolist() == cloned.columns.tolist()
    assert df.index.tolist() == cloned.index.tolist()


@pytest.mark.parametrize(
    "expect_multi_lvl, data",
    [
        (False, {"col1": [0, 1, 2, 3], "col2": [0, 1, 2, 3]}),
        (True, {("ft1", "col1"): [0, 1, 2, 3], ("ft1", "col2"): [0, 1, 2, 3]}),
        (True, {"ft1": {"col1": [0, 1, 2]}, "ft2": {"col1": [0, 1, 2]}}),
        (False, [[0, 1, 2], [0, 1, 2]]),
    ],
)
def test_dataframe_to_from_dict(expect_multi_lvl: bool, data: dict):
    """
    Creating dataframes from various raw data structures should have determined behavior
    such as not creating MultiIndex columns with a dict of simple key to array mappings.
    """
    df = server_utils.dataframe_from_dict(data)
    if expect_multi_lvl:
        assert isinstance(df.columns, pd.MultiIndex)
    else:
        assert not isinstance(df.columns, pd.MultiIndex)


@pytest.mark.parametrize(
    "index",
    (
        range(10),
        map(str, range(10)),
        map(str, random.sample(range(10), 10)),
        pd.date_range(start="2020-01-01", end="2020-01-02", periods=10),
        pd.date_range(start="2020-01-01", end="2020-01-02", periods=10).astype(str),
        random.sample(
            pd.date_range(start="2020-01-01", end="2020-01-02", periods=10).tolist(), 10
        ),
        random.sample(
            pd.date_range(start="2020-01-01", end="2020-01-02", periods=10)
            .astype(str)
            .tolist(),
            10,
        ),
    ),
)
def test_dataframe_from_dict_ordering(index):
    """
    We expect that from_dict should order based on the index, and will parse the index
    either as datetime or integers and sort in ascending order from there.
    """
    df = pd.DataFrame(np.random.random((10, 5)))
    df.index = index
    original = df.copy()

    # What we want
    if isinstance(original.index[0], str):
        # Parse as datetime or integers if index is string
        try:
            original.index = original.index.map(dateutil.parser.isoparse)
        except ValueError:
            original.index = original.index.map(int)
    original.sort_index(inplace=True)

    # What we get
    df_out = server_utils.dataframe_from_dict(server_utils.dataframe_to_dict(df))

    assert np.alltrue(df_out.index == original.index)
    assert np.alltrue(df_out.values == original.values)
