# -*- coding: utf-8 -*-

from typing import List, Union

import flask
import pytest
import pandas as pd
import numpy as np

from gordo_components.server import utils


@pytest.mark.parametrize(
    "data,tags,expect_success",
    [
        # Tags match length of data keys and data can be parsed; even though column names don't match tag names
        ({"c1": [1, 2, 3], "c2": [3, 2, 1]}, ["t1", "t2"], True),
        # Tags match length of data keys and data can be parsed.
        ([{"t1": 0, "t2": 1}, {"t1": 1, "t2": 2}], ["t1", "t2"], True),
        # Extra keys not matching tags should be ok, (trimmed out).
        ([{"t1": 0, "t2": 1}, {"t1": 1, "t2": 2}], ["t1"], True),
        # Column names aren't supplied in the dataframe data, tags should be applied as column names
        ([{0: 0, 1: 1}, {0: 1, 1: 2}], ["t1", "t2"], True),
        # If keys aren't all the keys in tags then return a response
        ([{"t1": 0, "t2": 1}, {"t1": 1, "t2": 2}], ["t1", "t2", "t3"], False),
        # Can't parse this data into a dataframe.
        ("Hello, I'm bad data :)", ["t1"], False),
    ],
)
def test_dataframe_from_dict(data: dict, tags: List[str], expect_success: bool):
    """
    Data can get into the server in a number of ways, basically any format
    that pandas.DataFrame.from_dict() supports. It opens for good flexibility
    we just need to test the function it is used in will 'fail' correctly or
    will reassign column names to tag names if needed.
    """
    app = flask.Flask(__name__)

    with app.app_context():
        result: Union[flask.Response, pd.DataFrame] = utils.dataframe_from_dict(
            data, tags, name="TEST"
        )

    if expect_success:

        assert isinstance(result, pd.DataFrame)

        # Check expected column names, if equal, or subset of tags
        if len(result.columns) == len(tags):
            assert result.columns.tolist() == tags
        else:
            assert all(col in tags for col in result.columns)
    else:

        # If it's not a dataframe it should be a response.
        assert isinstance(result, flask.Response)

        # Return a failed client request
        assert 400 <= result.status_code <= 499

        # Should have the name of the data being parsed in the error message
        assert "TEST" in result.data.decode()
