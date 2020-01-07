# -*- coding: utf-8 -*-

import pytest
import numpy as np
import pandas as pd

from gordo.machine.model import utils as model_utils
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


def test_metrics_wrapper():
    # make the features in y be in different scales
    y = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]) * [1, 100]

    # With no scaler provided it is relevant which of the two series gets an 80% error
    metric_func_noscaler = model_utils.metric_wrapper(mean_squared_error)

    mse_feature_one_wrong = metric_func_noscaler(y, y * [0.8, 1])
    mse_feature_two_wrong = metric_func_noscaler(y, y * [1, 0.8])

    assert not np.isclose(mse_feature_one_wrong, mse_feature_two_wrong)

    # With a scaler provided it is not relevant which of the two series gets an 80%
    # error
    scaler = MinMaxScaler().fit(y)
    metric_func_scaler = model_utils.metric_wrapper(mean_squared_error, scaler=scaler)

    mse_feature_one_wrong = metric_func_scaler(y, y * [0.8, 1])
    mse_feature_two_wrong = metric_func_scaler(y, y * [1, 0.8])

    assert np.isclose(mse_feature_one_wrong, mse_feature_two_wrong)


@pytest.mark.parametrize(
    "dates", [pd.date_range("2016-01-01", "2016-01-02", periods=10), None]
)
@pytest.mark.parametrize("tags", [["tag1", "tag2"], ["tag"]])
@pytest.mark.parametrize(
    "target_tag_list",
    (
        ["tag1", "tag2"],
        ["tag3", "tag4"],
        ["tagA"],
        ["tag1"],
        ["tagA", "tagB", "tagC"],
        None,
    ),
)
@pytest.mark.parametrize("output_offset", (0, 1, 2, 3))
def test_base_dataframe_creation(dates, tags, target_tag_list, output_offset):

    # Make model input based on tags
    size = len(dates if dates is not None else list(range(10))) * len(tags)
    model_input = np.random.random(size=size).reshape(-1, len(tags))

    # Model output based on target_tag_list
    size = len(dates if dates is not None else list(range(10))) * len(
        target_tag_list or list(range(20))
    )
    model_output = np.random.random(size=size).reshape((len(model_input), -1))

    # simulate where model's output length doesn't match it's input length
    # ie. as with an LSTM which outputs the offset of it's lookback window
    model_output = model_output[output_offset:]

    # pass in the arrays, of which model output's may be different lengths / shapes than the input
    # but should provide a valid dataframe in all cases.
    df = model_utils.make_base_dataframe(
        tags=tags,
        model_input=model_input,
        model_output=model_output,
        target_tag_list=target_tag_list,
        index=dates,
    )

    # offset column's like 'original-input' since it will be offsetted inside make_base_dataframe()
    assert np.array_equal(df["model-input"].values, model_input[-len(df) :, :])

    # Model input should always have column labels equal to the tags given
    assert df["model-input"].columns.tolist() == tags

    # Ensure model output matches
    assert np.array_equal(df["model-output"].values, model_output[-len(df) :, :])

    # Expected second level column names:
    # If target tags are defined, those should be the names
    if target_tag_list is not None:
        assert target_tag_list == df["model-output"].columns.tolist()

    # If they aren't defined, but model output shape matches input shape, tags should be the names.
    elif model_output.shape[1] == len(tags):
        assert tags == df["model-output"].columns.tolist()

    # Otherwise, column names should be simple range of feature length.
    else:
        assert (
            list(map(str, range(model_output.shape[1])))
            == df["model-output"].columns.tolist()
        )

    # Test expected index if dates were supplied or not
    if dates is not None:
        assert np.array_equal(df.index.values, dates.values[output_offset:])
    else:
        assert np.array_equal(df.index.values, np.arange(0, len(df)))
