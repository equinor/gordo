# -*- coding: utf-8 -*-

import logging
import functools
import timeit
import dateutil
from datetime import datetime
from typing import Union, List

import pandas as pd

from flask import request, g, jsonify, make_response, Response, current_app

from gordo_components.dataset.datasets import TimeSeriesDataset


"""
Tools used between different views

Namely :func:`.extract_X_y` and :func:`.base_dataframe` decorators which, when
wrapping methods will perform the tasks to extracting X & y, and creating the 
basic dataframe output, respectively.
"""

logger = logging.getLogger(__name__)


def dataframe_to_dict(df: pd.DataFrame) -> dict:
    """
    Convert a dataframe can have a :class:`pandas.MultiIndex` as columns into a dict
    where each key is the top level column name, and the value is the array
    of columns under the top level name. If it's a simple dataframe, :meth:`pandas.core.DataFrame.to_dict`
    will be used.

    This allows :func:`json.dumps` to be performed, where :meth:`pandas.DataFrame.to_dict()`
    would convert such a multi-level column dataframe into keys of ``tuple`` objects, which are
    not json serializable. However this ends up working with :meth:`pandas.DataFrame.from_dict`

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe expected to have columns of type :class:`pandas.MultiIndex` 2 levels deep.

    Returns
    -------
    List[dict]
        List of records representing the dataframe in a 'flattened' form.


    Examples
    --------
    >>> import pprint
    >>> import pandas as pd
    >>> import numpy as np
    >>> columns = pd.MultiIndex.from_tuples((f"feature{i}", f"sub-feature-{ii}") for i in range(2) for ii in range(2))
    >>> index = pd.date_range('2019-01-01', '2019-02-01', periods=2)
    >>> df = pd.DataFrame(np.arange(8).reshape((2, 4)), columns=columns, index=index)
    >>> df  # doctest: +NORMALIZE_WHITESPACE
                    feature0                    feature1
               sub-feature-0 sub-feature-1 sub-feature-0 sub-feature-1
    2019-01-01             0             1             2             3
    2019-02-01             4             5             6             7
    >>> serialized = dataframe_to_dict(df)
    >>> pprint.pprint(serialized)
    {'feature0': {'sub-feature-0': {'2019-01-01': 0, '2019-02-01': 4},
                  'sub-feature-1': {'2019-01-01': 1, '2019-02-01': 5}},
     'feature1': {'sub-feature-0': {'2019-01-01': 2, '2019-02-01': 6},
                  'sub-feature-1': {'2019-01-01': 3, '2019-02-01': 7}}}

    """
    # Need to copy, because Python's mutability allowed .index assignment to mutate the passed df
    data = df.copy()
    if isinstance(data.index, pd.DatetimeIndex):
        data.index = data.index.astype(str)
    if isinstance(df.columns, pd.MultiIndex):
        return {
            col: data[col].to_dict()
            if isinstance(data[col], pd.DataFrame)
            else pd.DataFrame(data[col]).to_dict()
            for col in data.columns.get_level_values(0)
        }
    else:
        return data.to_dict()


def dataframe_from_dict(data: dict) -> pd.DataFrame:
    """
    The inverse procedure done by :func:`.multi_lvl_column_dataframe_from_dict`
    Reconstructed a MultiIndex column dataframe from a previously serialized one.

    Expects ``data`` to be a nested dictionary where each top level key has a value
    capable of being loaded from :func:`pandas.core.DataFrame.from_dict`

    Parameters
    ----------
    data: dict
        Data to be loaded into a MultiIndex column dataframe

    Returns
    -------
    pandas.core.DataFrame
        MultiIndex column dataframe.

    Examples
    --------
    >>> serialized = {
    ... 'feature0': {'sub-feature-0': {'2019-01-01': 0, '2019-02-01': 4},
    ...              'sub-feature-1': {'2019-01-01': 1, '2019-02-01': 5}},
    ... 'feature1': {'sub-feature-0': {'2019-01-01': 2, '2019-02-01': 6},
    ...              'sub-feature-1': {'2019-01-01': 3, '2019-02-01': 7}}
    ... }
    >>> dataframe_from_dict(serialized)  # doctest: +NORMALIZE_WHITESPACE
                    feature0                    feature1
           sub-feature-0 sub-feature-1 sub-feature-0 sub-feature-1
    2019-01-01             0             1             2             3
    2019-02-01             4             5             6             7
    """

    if isinstance(data, dict) and any(isinstance(val, dict) for val in data.values()):
        try:
            keys = data.keys()
            df: pd.DataFrame = pd.concat(
                (pd.DataFrame.from_dict(data[key]) for key in keys), axis=1, keys=keys
            )
        except (ValueError, AttributeError):
            df = pd.DataFrame.from_dict(data)
    else:
        df = pd.DataFrame.from_dict(data)

    try:
        df.index = df.index.map(dateutil.parser.isoparse)  # type: ignore
    except (TypeError, ValueError):
        logger.debug("Could not parse index to pandas.DatetimeIndex")
        pass  # It wasn't a datetime index after all, no worries.
    return df


def parse_iso_datetime(datetime_str: str) -> datetime:
    parsed_date = dateutil.parser.isoparse(datetime_str)  # type: ignore
    if parsed_date.tzinfo is None:
        raise ValueError(
            f"Provide timezone to timestamp {datetime_str}."
            f" Example: for UTC timezone use {datetime_str + 'Z'} or {datetime_str + '+00:00'} "
        )
    return parsed_date


def _verify_dataframe(
    df: pd.DataFrame, expected_columns: List[str]
) -> Union[Response, pd.DataFrame]:
    """
    Verify the dataframe, setting the column names to ``expected_columns``
    if not already labeled and the length of the columns match the length of the expected columns.

    If it fails, it will return an instance of :class:`flask.wrappers.Response`

    Parameters
    ----------
    df: pandas.core.DataFrame
        DataFrame to verify.
    expected_columns: List[str]
        List of expected column names to give if the dataframe does not consist of them
        but the number of columns matches ``len(expected_columns)``

    Returns
    -------
    Union[flask.wrappers.Response, pandas.core.DataFrame]
    """
    if not isinstance(df.columns, pd.MultiIndex):
        if not all(col in df.columns for col in expected_columns):

            # If the length doesn't mach, then we can't reliably determine what data we have hre.
            if len(df.columns) != len(expected_columns):
                msg = dict(
                    message=f"Unexpected features: "
                    f"was expecting {expected_columns} length of {len(expected_columns)}, "
                    f"but got {df.columns} length of {len(df.columns)}"
                )
                return make_response((jsonify(msg), 400))

            # Otherwise we were send a list/ndarray data format which we assume the client has
            # ordered correctly to the order of the expected_columns.
            else:
                df.columns = expected_columns

        # All columns exist in the dataframe, select them which thus ensures order and removes extra columns
        else:
            df = df[expected_columns]
        return df
    else:
        msg = {
            "message": f"Server does not support multi-level dataframes at this time: {df.columns.tolist()}"
        }
        return make_response((jsonify(msg), 400))


def extract_X_y(method):
    """
    For a given flask view, will attempt to extract an 'X' and 'y' from
    the request and assign it to flask's 'g' global request context

    If it fails to extract 'X' and (optionally) 'y' from the request, it will **not** run the
    function but return a ``BadRequest`` response notifying the client of the failure.

    Parameters
    ----------
    method: Callable
        The flask route to decorate, and will return it's own response object
        and will want to use ``flask.g.X`` and/or ``flask.g.y``

    Returns
    -------
    flask.Response
        Will either run a :class:`flask.Response` with status code 400 if it fails
        to extract the X and optionally the y. Otherwise will run the decorated ``method``
        which is also expected to return some sort of :class:`flask.Response` object.
    """

    @functools.wraps(method)
    def wrapper_method(self, *args, **kwargs):

        # Data provided by the client
        if request.method == "POST":
            X = request.json.get("X")
            y = request.json.get("y")

            if X is None:
                message = dict(message='Cannot predict without "X"')
                return make_response((jsonify(message), 400))

            # Convert X and (maybe) y into dataframes.
            X = dataframe_from_dict(X)
            X = _verify_dataframe(X, [t.name for t in self.tags])

            # Y is ok to be None for BaseView, view(s) like Anomaly might require it.
            if y is not None:
                y = dataframe_from_dict(y)
                y = _verify_dataframe(y, [t.name for t in self.target_tags])

            # If either X or y came back as a Response type, there was an error
            for data_or_resp in [X, y]:
                if isinstance(data_or_resp, Response):
                    return data_or_resp

        # Data must be queried from Influx given dates passed in request.
        elif request.method == "GET":

            params = request.get_json() or request.args

            if not all(k in params for k in ("start", "end")):
                message = dict(
                    message="must provide iso8601 formatted dates with timezone-information for parameters 'start' and 'end'"
                )
                return make_response((jsonify(message), 400))

            # Extract the dates from parameters
            try:
                start = parse_iso_datetime(params["start"])
                end = parse_iso_datetime(params["end"])
            except ValueError:
                logger.error(
                    f"Failed to parse start and/or end date to ISO: start: "
                    f"{params['start']} - end: {params['end']}"
                )
                message = dict(
                    message="Could not parse start/end date(s) into ISO datetime. must provide iso8601 formatted dates for both."
                )
                return make_response((jsonify(message), 400))

            # Make request time span of one day
            if (end - start).days:
                message = dict(
                    message="Need to request a time span less than 24 hours."
                )
                return make_response((jsonify(message), 400))

            logger.debug("Fetching data from data provider")
            before_data_fetch = timeit.default_timer()
            dataset = TimeSeriesDataset(
                data_provider=g.data_provider,
                from_ts=start - self.frequency.delta,
                to_ts=end,
                resolution=current_app.metadata["dataset"]["resolution"],
                tag_list=self.tags,
                target_tag_list=self.target_tags or None,
            )
            X, y = dataset.get_data()
            logger.debug(
                f"Fetching data from data provider took "
                f"{timeit.default_timer()-before_data_fetch} seconds"
            )
            # Want resampled buckets equal or greater than start, but less than end
            # b/c if end == 00:00:00 and req = 10 mins, a resampled bucket starting
            # at 00:00:00 would imply it has data until 00:10:00; which is passed
            # the requested end datetime
            X = X[
                (X.index > start - self.frequency.delta)
                & (X.index + self.frequency.delta < end)
            ]

            # TODO: Remove/rework this once we add target_tags assignments in workflow generator for autoencoders.
            if y is None:
                y = X.copy()
            else:
                y = y.loc[X.index]

        else:
            raise NotImplementedError(
                f"Cannot extract X and y from '{request.method}' request."
            )

        # Assign X and y to the request's global context
        g.X, g.y = X, y

        # And run the original method.
        return method(self, *args, **kwargs)

    return wrapper_method
