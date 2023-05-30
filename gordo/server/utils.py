# -*- coding: utf-8 -*-

import logging
import functools
import zlib
import os
import io
import pickle
import re
import shutil

import dateutil
import timeit
from datetime import datetime
from typing import Union, List

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from flask import request, g, jsonify, make_response, Response
from functools import lru_cache, wraps
from sklearn.base import BaseEstimator
from werkzeug.exceptions import NotFound, UnprocessableEntity, InternalServerError

from gordo import serializer

from .properties import get_tags, get_target_tags

"""
Tools used between different views

Namely :func:`.extract_X_y` and :func:`.base_dataframe` decorators which, when
wrapping methods will perform the tasks to extracting X & y, and creating the 
basic dataframe output, respectively.
"""

logger = logging.getLogger(__name__)

gordo_name_re = re.compile(r"^[a-zA-Z\d-]+")
revision_re = re.compile(r"^\d+$")


def validate_revision(revision: str) -> bool:
    return bool(revision_re.match(revision))


def dataframe_into_parquet_bytes(
    df: pd.DataFrame, compression: str = "snappy"
) -> bytes:
    """
    Convert a dataframe into bytes representing a parquet table.

    Parameters
    ----------
    df
        DataFrame to be compressed
    compression
        Compression to use, passed to  :func:`pyarrow.parquet.write_table`

    Returns
    -------
    """
    table = pa.Table.from_pandas(df)
    buf = pa.BufferOutputStream()
    pq.write_table(table, buf, compression=compression)
    return buf.getvalue().to_pybytes()


def dataframe_from_parquet_bytes(buf: bytes) -> pd.DataFrame:
    """
    Convert bytes representing a parquet table into a pandas dataframe.

    Parameters
    ----------
    buf
        Bytes representing a parquet table. Can be the direct result from
        `func`::gordo.server.utils.dataframe_into_parquet_bytes

    Returns
    -------
    """
    table = pq.read_table(io.BytesIO(buf))
    return table.to_pandas()


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
    df
        Dataframe expected to have columns of type :class:`pandas.MultiIndex` 2 levels deep.

    Returns
    -------
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
    data
        Data to be loaded into a MultiIndex column dataframe

    Returns
    -------
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
        df.index = df.index.map(int)

    df.sort_index(inplace=True)

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
    df
        DataFrame to verify.
    expected_columns
        List of expected column names to give if the dataframe does not consist of them
        but the number of columns matches ``len(expected_columns)``

    Returns
    -------
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
    method
        The flask route to decorate, and will return it's own response object
        and will want to use ``flask.g.X`` and/or ``flask.g.y``

    Returns
    -------
        Will either run a :class:`flask.Response` with status code 400 if it fails
        to extract the X and optionally the y. Otherwise will run the decorated ``method``
        which is also expected to return some sort of :class:`flask.Response` object.
    """

    @functools.wraps(method)
    def wrapper_method(*args, **kwargs):
        start_time = timeit.default_timer()
        # Data provided by the client
        if request.method == "POST":

            # Always require an X, be it in JSON or file/parquet format.
            if request.is_json:
                if "X" not in (request.json or {}):
                    message = dict(message='Cannot predict without "X"')
                    return make_response((jsonify(message), 400))
            else:
                if "X" not in request.files:
                    message = dict(message='Cannot predict without "X"')
                    return make_response((jsonify(message), 400))

            if request.is_json:
                X = dataframe_from_dict(request.json["X"])
                y = request.json.get("y")
                if y is not None:
                    y = dataframe_from_dict(y)
            else:
                X = dataframe_from_parquet_bytes(request.files["X"].read())
                y = request.files.get("y")
                if y is not None:
                    y = dataframe_from_parquet_bytes(y.read())

            X = _verify_dataframe(X, [t.name for t in get_tags()])

            # Verify y if it's not None
            if y is not None:
                y = _verify_dataframe(y, [t.name for t in get_target_tags()])

            # If either X or y came back as a Response type, there was an error
            for data_or_resp in [X, y]:
                if isinstance(data_or_resp, Response):
                    return data_or_resp
        else:
            raise NotImplementedError(
                f"Cannot extract X and y from '{request.method}' request."
            )

        # Assign X and y to the request's global context
        g.X, g.y = X, y

        try:
            logger.debug(f"Size of X: {X.size}, size of y: {y.size}")
        except AttributeError:
            logger.debug(f"Size of X: {X.size}, y is None")
        logger.debug(f"Time to parse X and y: {timeit.default_timer() - start_time}s")

        # And run the original method.
        return method(*args, **kwargs)

    return wrapper_method


@lru_cache(maxsize=int(os.getenv("N_CACHED_MODELS", 2)))
def load_model(directory: str, name: str) -> BaseEstimator:
    """
    Load a given model from the directory by name.

    Parameters
    ----------
    directory
        Directory to look for the model
    name
        Name of the model to load, this would be the sub directory within the
        directory parameter.

    Returns
    -------
    """
    start_time = timeit.default_timer()
    model = serializer.load(os.path.join(directory, name))
    logger.debug(f"Time to load model: {timeit.default_timer() - start_time}s")
    return model


def check_metadata_file(directory: str, name: str):
    """
    Checking if the directory with metadata exists since it might be deleted through DELETE endpoint
    """
    # TODO consider using https://pypi.org/project/ring/ with the ability to delete items from lru cache
    full_model_dir = os.path.join(directory, name)
    if not serializer.metadata_path(full_model_dir):
        raise FileNotFoundError("Unable to load metadata.json file")


def load_metadata(directory: str, name: str) -> dict:
    """
    Load metadata from a directory for a given model by name.

    Parameters
    ----------
    directory
        Directory to look for the model's metadata
    name
        Name of the model to load metadata for, this would be the sub directory
        within the directory parameter.

    Returns
    -------
    """
    compressed_metadata = _load_compressed_metadata(directory, name)
    return pickle.loads(zlib.decompress(compressed_metadata))


_n_cached_metadata = int(os.getenv("N_CACHED_METADATA", 250))


@lru_cache(maxsize=_n_cached_metadata)
def _load_compressed_metadata(directory: str, name: str):
    """
    Loads the metadata for model 'name' from directory 'directory', and returns it as a
    zlib compressed pickle, to use as little space as possible in the cache.

    Notes
    ----
    Some simple measurement indicated that a typical metadata dict uses 37kb in memory,
    while pickled it uses 8kb, and pickled-compressed it uses 4kb.

    """
    metadata = serializer.load_metadata(os.path.join(directory, name))
    return zlib.compress(pickle.dumps(metadata))


def delete_revision(directory: str, name: str):
    """
    Delete model revision

    Parameters
    ----------
    directory - Revision directory
    name - Model name
    """
    full_path = os.path.join(directory, name)
    if not os.path.isfile(os.path.join(full_path, "metadata.json")):
        raise NotFound("Not found")
    shutil.rmtree(full_path, ignore_errors=True)
    if os.path.exists(full_path):
        raise InternalServerError("Unable to delete this model revision folder")
    if not os.listdir(directory):
        shutil.rmtree(directory, ignore_errors=True)
        if os.path.exists(directory):
            raise InternalServerError("Unable to delete this revision folder")


def validate_gordo_name(gordo_name: str):
    """
    gordo_name argument should contains alpha-numericals or '-' symbols
    """
    if gordo_name and not gordo_name_re.match(gordo_name):
        raise UnprocessableEntity("gordo_name field has wrong format")


@lru_cache(maxsize=_n_cached_metadata)
def load_info(directory: str, name: str) -> dict:
    # TODO better docstring
    return serializer.load_info(os.path.join(directory, name))


def metadata_required(f):
    """
    Decorate a view which has ``gordo_name`` as a url parameter and will
    set ``g.metadata`` to that model's metadata
    """

    @wraps(f)
    def wrapper(*args: tuple, gordo_project: str, gordo_name: str, **kwargs: dict):
        validate_gordo_name(gordo_name)
        g.info = {}
        try:
            g.info = load_info(directory=g.collection_dir, name=gordo_name)
        except FileNotFoundError:
            pass
        try:
            check_metadata_file(g.collection_dir, gordo_name)
            g.metadata = load_metadata(directory=g.collection_dir, name=gordo_name)
        except FileNotFoundError:
            raise NotFound(f"No metadata found for '{gordo_name}'")
        else:
            return f(*args, **kwargs)

    return wrapper


def model_required(f):
    """
    Decorate a view which has ``gordo_name`` as a url parameter and will
    set ``g.model`` to be the loaded model and ``g.metadata``
    to that model's metadata
    """

    @wraps(f)
    def wrapper(*args: tuple, gordo_project: str, gordo_name: str, **kwargs: dict):
        validate_gordo_name(gordo_name)
        try:
            check_metadata_file(g.collection_dir, gordo_name)
            g.model = load_model(directory=g.collection_dir, name=gordo_name)
        except FileNotFoundError:
            raise NotFound(f"No such model found: '{gordo_name}'")
        else:
            # If the model was required, the metadata is also required.
            return metadata_required(f)(
                *args, gordo_project=gordo_project, gordo_name=gordo_name, **kwargs
            )

    return wrapper
