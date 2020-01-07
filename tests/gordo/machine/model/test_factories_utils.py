# -*- coding: utf-8 -*-

import pytest

from gordo.machine.model.factories.utils import hourglass_calc_dims, check_dim_func_len


@pytest.mark.parametrize(
    "test_input,test_expected",
    [
        ((0.2, 4, 5), (4, 3, 2, 1)),
        ((0.5, 3, 10), (8, 7, 5)),
        ((0.5, 3, 3), (3, 2, 2)),
        ((0.3, 3, 10), (8, 5, 3)),
        ((1, 3, 10), (10, 10, 10)),
        ((0, 3, 100000), (66667, 33334, 1)),
    ],
)
def test_hourglass_calc_dims_check_dims(test_input, test_expected):
    """
    Test that hourglass_calc_dims implements the correct dimensions
    """
    dims = hourglass_calc_dims(*test_input)
    assert dims == test_expected


def test_check_dim_func_len():
    """
    Test that error is raised if encoding/decoding parameters are not equal of length
    """
    with pytest.raises(ValueError):
        check_dim_func_len("test", dim=(256, 128), func=("tanh", "tanh", "tanh"))

    with pytest.raises(ValueError):
        check_dim_func_len("test", dim=(256, 128, 56), func=("tanh", "tanh"))
