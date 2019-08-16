# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from gordo_components.server import utils as server_utils


def test_multi_lvl_column_dataframe_from_to_dict():

    columns = pd.MultiIndex.from_product((("feature1", "feature2"), ("col1", "col2")))
    df = pd.DataFrame(
        np.random.random((10, 4)),
        columns=columns,
        index=pd.date_range("2016-01-01", "2016-02-01", periods=10),
    )

    assert isinstance(df.index, pd.DatetimeIndex)

    cloned = server_utils.multi_lvl_column_dataframe_from_dict(
        server_utils.multi_lvl_column_dataframe_to_dict(df)
    )

    # Ensure the function hasn't mutated the index.
    assert isinstance(df.index, pd.DatetimeIndex)

    assert np.allclose(df.values, cloned.values)
    assert df.columns.tolist() == cloned.columns.tolist()
