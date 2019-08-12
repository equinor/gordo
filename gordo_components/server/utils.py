# -*- coding: utf-8 -*-

import logging
import functools
import timeit
import traceback
import typing
import dateutil
from datetime import datetime

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


def dataframe_from_dict(
    data: dict, tags: typing.List[str], name: str
) -> typing.Union[pd.DataFrame, Response]:
    """
    Convert data supported by :meth:`pandas.DataFrame.from_dict` into a DataFrame
    and ensure the columns are the ``tags``. Will return a :class:`flask.Response` if
    parsing of the data into a DataFrame failed.

    Parameters
    ----------
    df: Union[dict, list]
        Any data supported by :py:meth:`pandas.DataFrame.from_dict`
    tags: List[str]
        The expected column names for the dataframe. If data supplied does not
        have column names, but the length of those columns matches the tags, these
        tags will be used as column names.
    name: str
        How to format the :class:`flask.Response` message when referring to the data supplied.
        This will only be used if parsing the data into a DataFrame object fails.

    Returns
    -------
    Union[pandas.DataFrame, flask.Response]
    """
    # Parse X and possibly y into a dataframe
    try:
        df = pd.DataFrame.from_dict(data)
    except ValueError:
        return make_response((jsonify(dict(message=f"Unable to parse '{name}'")), 400))
    else:
        # All tags are in the columns, so we'll take the subset to match the tags.
        # **this also ensures the order of the columns is the order of the tags**
        if all(tag in df.columns for tag in tags):
            df = df[tags]

        # n_features matches, set columns to be tag names
        # This is the case that column names were not supplied, such as 'bare'
        # nested arrays.
        elif len(df.columns) == len(tags):
            df.columns = tags

        # Size isn't what was expected either, we have no idea what this is.
        else:
            message = dict(
                message=f"Expected feature names/length of: {tags}/{len(tags)} in {name} "
                f"but found {df.columns}/{len(df.columns)}"
            )
            return make_response((jsonify(message), 400))
        return df


def multi_lvl_column_dataframe_to_dict(df: pd.DataFrame) -> dict:
    """
    Convert a dataframe which has a :class:`pandas.MultiIndex` as columns into a dict
    where each key is the top level column name, and the value is the array
    of columns under the top level name.

    This allows :func:`json.dumps` to be performed, where :meth:`pandas.DataFrame.to_dict()`
    would convert such a multi-level column dataframe into keys of ``tuple`` objects, which are
    not json serializable. However this ends up working with :meth:`pandas.DataFrame.from_dict`

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe expected to have columns of type :class:`pandas.MultiIndex`

    Returns
    -------
    List[dict]
        List of records representing the dataframe in a 'flattened' form.


    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> columns = pd.MultiIndex.from_tuples((f"feature{i}", f"sub-feature-{ii}") for i in range(2) for ii in range(2))
    >>> df = pd.DataFrame(np.arange(8).reshape((2, 4)), columns=columns)
    >>> df  # doctest: +NORMALIZE_WHITESPACE
           feature0                    feature1
      sub-feature-0 sub-feature-1 sub-feature-0 sub-feature-1
    0             0             1             2             3
    1             4             5             6             7
    >>> multi_lvl_column_dataframe_to_dict(df)
    [{'feature0': [0, 1], 'feature1': [2, 3]}, {'feature0': [4, 5], 'feature1': [6, 7]}]
    """

    # Note: It is possible to do this more simply with nested dict comprehension,
    # but ends up making it ~5x slower.

    # This gets the 'top level' names of the multi level column names
    # it will contain names like 'model-output' and then sub column(s)
    # of the actual model output.
    names = df.columns.get_level_values(0).unique()

    # Now a series where each row has an index with the name of the feature
    # which corresponds to 'names' above.
    records = (
        # Stack the dataframe so second level column names become second level indexs
        df.stack()
        # For each column now, unstack the previous second level names (which are now the indexes of the series)
        # back into a dataframe with those names, and convert to list; if it's a Series we'll need to reshape it
        .apply(
            lambda col: col.reindex(df[col.name].columns, level=1)
            .unstack()
            .dropna(axis=1)
            .values.tolist()
            if isinstance(df[col.name], pd.DataFrame)
            else df[col.name].values.reshape(-1, 1).tolist()
        )
    )

    results: typing.List[dict] = []

    for i, name in enumerate(names):

        # For each top level name, we'll select its column, unstack so that
        # previous second level names moved into the index will then be column
        # names again, and convert that to a list, matched to the name.
        values = map(lambda row: {name: row}, records[name])

        # If we have results, we'll update the record/row data with this
        # current name. ie {'col1': [1, 2}} -> {'col1': [1, 2], 'col2': [3, 4]}
        # if the current column name is 'col2' and values [3, 4] for the first row,
        # and so on.
        if i == 0:
            results = list(values)
        else:
            [rec.update(d) for rec, d in zip(results, values)]

    return results


def parse_iso_datetime(datetime_str: str) -> datetime:
    parsed_date = dateutil.parser.isoparse(datetime_str)  # type: ignore
    if parsed_date.tzinfo is None:
        raise ValueError(
            f"Provide timezone to timestamp {datetime_str}."
            f" Example: for UTC timezone use {datetime_str + 'Z'} or {datetime_str + '+00:00'} "
        )
    return parsed_date


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

            if request.json is not None:
                X = request.json.get("X")
                y = request.json.get("y")

                if X is None:

                    message = dict(message='Cannot predict without "X"')
                    return make_response((jsonify(message), 400))

                # Convert X and (maybe) y into dataframes.
                X = dataframe_from_dict(
                    X, tags=list(tag.name for tag in self.tags), name="X"
                )

                # Y is ok to be None for BaseView, view(s) like Anomaly might require it.
                if y is not None and self.target_tags:
                    y = dataframe_from_dict(
                        y, list(tag.name for tag in self.target_tags), name="y"
                    )

                # If either X or y came back as a Response type, there was an error
                for data_or_resp in [X, y]:
                    if isinstance(data_or_resp, Response):
                        return data_or_resp
            else:
                try:
                    data = pd.read_msgpack(request.data)
                except Exception as exc:
                    logger.error(f"Unable to parse msgpack: {traceback.format_exc()}")
                    msg = {"message": "Unable to parse msgpack data"}
                    return make_response((jsonify(msg), 400))
                else:
                    # TODO: Catch potential key error(s)
                    X = data[list(tag.name for tag in self.tags)]
                    y = data[list(tag.name for tag in self.target_tags)]

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
