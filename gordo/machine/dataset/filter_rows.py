import logging
import pandas as pd
import numpy as np
from typing import Union

logger = logging.getLogger(__name__)


def apply_buffer(mask: pd.Series, buffer_size: int = 0):
    """
    Take a mask (boolean series) where True indicates keeping a value, and False
    represents removing the value. This will 'expand' those indexes marked as `False`
    to the symmetrical bounds of ``buffer_size``

    Parameters
    ----------
    mask: pandas.core.Series
        Boolean pandas series
    buffer_size: int
        Size to buffer around ``False`` values

    Examples
    --------
    >>> import pandas as pd
    >>> series = pd.Series([True, True, False, True, True])
    >>> series = apply_buffer(series, buffer_size=1)
    >>> series
    0     True
    1    False
    2    False
    3    False
    4     True
    dtype: bool

    Returns
    -------
    None
    """
    if (not any(mask)) or (buffer_size == 0):
        return mask

    array = 1 - np.array(mask.to_numpy(), dtype=float)
    kernel = np.ones(buffer_size * 2 + 1, dtype=float)

    if len(kernel) > len(array):
        mask.values[:] = False
        return mask

    ans = np.convolve(a=array, v=kernel, mode="same")
    mask.values[:] = ans < 1
    return mask


def pandas_filter_rows(
    df: pd.DataFrame, filter_str: Union[str, list], buffer_size: int = 0
):
    """ Filter pandas data frame based on list or string of conditions.

    Note:
    pd.DataFrame.eval of a list returns a numpy.ndarray and is limited to 100 list items.
    The sparse evaluation with numexpr pd.DataFrame.eval of a combined string logic, can only consist of
    a maximum 32 (current dependency) or 242 logical parts (latest release) and returns a pd.Series
    Therefore, list elements are evaluated in batches of n=15 (to be safe) and evaluate iterative.

    Parameters
    ----------
    df: pandas.Dataframe
      Dataframe to filter rows from. Does not modify the parameter
    filter_str: str or list
      String representing the filter. Can be a boolean combination of conditions,
      where conditions are comparisons of column names and either other columns
      or numeric values. The rows matching the filter are kept.
      Column names with spaces must be quoted with backticks,
      names without spaces could be quoted with backticks or be unquoted.
      Example of legal filters are " `Tag A` > 5 " , " (`Tag B` > 1) | (`Tag C` > 4)"
      '(`Tag D` < 5) ', " (TagB > 5) "
      The parameter can also be a list, in which the items will be joined by logical " & ".
    buffer_size: int
      Area fore and aft of the application of ``fitler_str`` to also mark for removal.

    Returns
    -------
    pandas.Dataframe
        The dataframe containing only rows matching the filter

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> df = pd.DataFrame(list(np.ndindex((3,3))), columns=list('AB'))
    >>> df
       A  B
    0  0  0
    1  0  1
    2  0  2
    3  1  0
    4  1  1
    5  1  2
    6  2  0
    7  2  1
    8  2  2
    >>> pandas_filter_rows(df, "`A`>1")
       A  B
    6  2  0
    7  2  1
    8  2  2
    >>> pandas_filter_rows(df, "`A`> B")
       A  B
    3  1  0
    6  2  0
    7  2  1
    >>> pandas_filter_rows(df, "(`A`>1) | (`B`<1)")
       A  B
    0  0  0
    3  1  0
    6  2  0
    7  2  1
    8  2  2
    >>> pandas_filter_rows(df, "(`A`>1) & (`B`<1)")
       A  B
    6  2  0
    >>> pandas_filter_rows(df, ["A>1", "B<1"])
       A  B
    6  2  0
    >>> pandas_filter_rows(df, ["A!=1", "B<3"])
       A  B
    0  0  0
    1  0  1
    2  0  2
    6  2  0
    7  2  1
    8  2  2
    >>> pandas_filter_rows(df, ["A!=1", "B<3"], buffer_size=1)
       A  B
    0  0  0
    1  0  1
    7  2  1
    8  2  2
    """
    logger.info("Applying numerical filtering to data of shape %s", df.shape)

    if isinstance(filter_str, str):
        mask = df.eval(filter_str)

    if isinstance(filter_str, list):
        mask = []
        for filter_i in _batch(iterable=filter_str, n=15):
            mask.append(df.eval(" & ".join(filter_i)))

        mask = pd.concat(mask, axis=1).all(axis=1)

    if buffer_size != 0:
        mask = apply_buffer(mask, buffer_size=buffer_size)

    df = df[list(mask)]
    logger.info("Shape of data after numerical filtering: %s", df.shape)
    return df


def _batch(iterable, n: int):
    """Helper function for creating batches on list items"""
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]
