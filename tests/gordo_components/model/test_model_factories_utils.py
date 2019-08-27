# -*- coding: utf-8 -*-

import pytest

from gordo_components.model.factories.utils import (
    hourglass_calc_dims,
    check_dim_func_len,
)


def test_hourglass_calc_dims_check_dims():
    """
    Test that hourglass_calc_dims implements the correct dimensions
    """

    dims = hourglass_calc_dims(0.2, 4, 5)
    assert dims == [4, 3, 2, 1]
    dims = hourglass_calc_dims(0.5, 3, 10)
    assert dims == [8, 7, 5]
    dims = hourglass_calc_dims(0.5, 3, 3)
    assert dims == [3, 2, 2]
    dims = hourglass_calc_dims(0.3, 3, 10)
    assert dims == [8, 5, 3]
    dims = hourglass_calc_dims(1, 3, 10)
    assert dims == [10, 10, 10]
    dims = hourglass_calc_dims(0, 3, 100000)
    assert dims == [66667, 33334, 1]


def test_check_dim_func_len():
    """
    Test that encoding/decoding dimension and function parameters are equal of length
    """
    # Raises a ValueError if the len of dim and func are not equal
    with pytest.raises(ValueError):
        check_dim_func_len("test", dim=(256, 128), func=("tanh", "tanh", "tanh"))

    with pytest.raises(ValueError):
        check_dim_func_len("test", dim=(256, 128, 56), func=("tanh", "tanh"))
