import pytest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from pandas.core.computation.ops import UndefinedVariableError
from gordo.machine.dataset.filter_rows import (
    pandas_filter_rows,
    parse_pandas_filter_vars,
    apply_buffer,
)


def test_parse_filter_vars():
    expr = "`a` > 0 & `c` == 3.0"
    result = set(parse_pandas_filter_vars(expr))
    assert result == {"c", "a"}

    expr = "a < 0 & c = 3.0"
    result = set(parse_pandas_filter_vars(expr))
    assert result == {"c", "a"}

    expr = "`var$' _name` > 22"
    result = set(parse_pandas_filter_vars(expr))
    assert result == {"var$' _name"}

    expr = "sin(col1) > 0.5 and cos(`col2`) < 0.5"
    result = set(parse_pandas_filter_vars(expr))
    assert result == {"col1", "col2"}

    expr = ["tag1 > 0", "tag2 < 100"]
    result = set(parse_pandas_filter_vars(expr))
    assert result == {"tag1", "tag2"}

    expr = "0 < index < 100"
    result = set(parse_pandas_filter_vars(expr))
    assert result == set()

    with pytest.raises(SyntaxError):
        expr = "`|tag` > 0"
        parse_pandas_filter_vars(expr)

    expr = "sin(col1) > 10 & 0 < index < 100"
    result = set(parse_pandas_filter_vars(expr, with_special_vars=True))
    assert result == {'sin', 'col1', 'index'}


def test_filter_rows_basic():
    df = pd.DataFrame(list(np.ndindex((10, 2))), columns=["Tag  1", "Tag 2"])
    assert len(pandas_filter_rows(df, "`Tag  1` <= `Tag 2`")) == 3
    assert len(pandas_filter_rows(df, "`Tag  1` == `Tag 2`")) == 2
    assert len(pandas_filter_rows(df, "(`Tag  1` <= `Tag 2`) | (`Tag 2` < 2)")) == 20
    assert len(pandas_filter_rows(df, "(`Tag  1` <= `Tag 2`) | (`Tag 2` < 0.9)")) == 12
    assert len(pandas_filter_rows(df, "(`Tag  1` > 0) & (`Tag 2` > 0)")) == 9
    assert len(pandas_filter_rows(df, ["`Tag  1` > 0", "`Tag 2` > 0"])) == 9

    assert_frame_equal(
        pandas_filter_rows(df, "(`Tag  1` <= `Tag 2`)"),
        pandas_filter_rows(df, "~(`Tag  1` > `Tag 2`)"),
    )


def test_filter_rows_catches_illegal():
    df = pd.DataFrame(list(np.ndindex((10, 2))), columns=["Tag  1", "Tag 2"])
    with pytest.raises(UndefinedVariableError):
        pandas_filter_rows(df, "sys.exit(0)")
    with pytest.raises(NotImplementedError):
        pandas_filter_rows(df, "lambda x:x")
    with pytest.raises(ValueError):
        pandas_filter_rows(df, "__import__('os').system('clear')"), ValueError


@pytest.mark.parametrize(
    "buffer_size,series,expected",
    [
        (
            1,
            pd.Series([1, 1, 1, 1, 0, 1, 1, 1, 1]),
            pd.Series([1, 1, 1, 0, 0, 0, 1, 1, 1]),
        ),
        (
            2,
            pd.Series([1, 1, 1, 1, 0, 1, 1, 1, 1]),
            pd.Series([1, 1, 0, 0, 0, 0, 0, 1, 1]),
        ),
        (
            1,
            pd.Series([0, 1, 1, 1, 1, 1, 1, 1, 0]),
            pd.Series([0, 0, 1, 1, 1, 1, 1, 0, 0]),
        ),
        (
            2,
            pd.Series([0, 0, 1, 0, 1, 1, 1, 1, 1]),
            pd.Series([0, 0, 0, 0, 0, 0, 1, 1, 1]),
        ),
    ],
)
def test_filter_rows_buffer(buffer_size, series, expected):
    series = series.astype(bool)
    expected = expected.astype(bool)

    apply_buffer(series, buffer_size=buffer_size)
    assert np.alltrue(np.equal(series, expected))
