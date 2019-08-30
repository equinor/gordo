# -*- coding: utf-8 -*-

import pytest
import pandas as pd
import numpy as np

from gordo_components.server import utils as server_utils


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
