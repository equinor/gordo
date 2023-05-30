# -*- coding: utf-8 -*-

import simplejson
import logging
import os
import re
import pickle

from typing import Union, Any, Optional  # pragma: no flakes

from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin, BaseEstimator  # noqa

from gordo.machine.model.base import GordoBase

logger = logging.getLogger(__name__)

N_STEP_REGEX = re.compile(r".*n_step=([0-9]+)")
CLASS_REGEX = re.compile(r".*class=(.*$)")


def dumps(model: Union[Pipeline, GordoBase]) -> bytes:
    """
    Dump a model into a bytes representation suitable for loading from
    ``gordo.serializer.loads``

    Parameters
    ----------
    model
        A gordo model/pipeline

    Returns
    -------
        Serialized model which supports loading via ``serializer.loads()``

    Example
    -------
    >>> from gordo.machine.model.models import KerasAutoEncoder
    >>> from gordo import serializer
    >>>
    >>> model = KerasAutoEncoder('feedforward_symmetric')
    >>> serialized = serializer.dumps(model)
    >>> assert isinstance(serialized, bytes)
    >>>
    >>> model_clone = serializer.loads(serialized)
    >>> assert isinstance(model_clone, KerasAutoEncoder)
    """
    return pickle.dumps(model)


def loads(bytes_object: bytes) -> GordoBase:
    """
    Load a GordoBase model from bytes dumped from ``gordo.serializer.dumps``

    Parameters
    ----------
    bytes_object
        Bytes to be loaded, should be the result of `serializer.dumps(model)`

    Returns
    -------
        Custom gordo model, scikit learn pipeline or other scikit learn like object.
    """
    return pickle.loads(bytes_object)


def metadata_path(
    source_dir: Union[os.PathLike, str]
) -> Optional[Union[os.PathLike, str]]:
    """
    Returns path to metadata.json file, if exists.

    """
    return _json_file_path(source_dir, "metadata.json")


def _json_file_path(source_dir: Union[os.PathLike, str], file_name: str):
    # Since this function can take the top level dir, or a dir directly
    # into the first step of the pipeline, we need to check both for the file
    possible_paths = [
        os.path.join(source_dir, file_name),
        os.path.join(source_dir, "..", file_name),
    ]
    return next((path for path in possible_paths if os.path.exists(path)), None)


def _load_json_file(source_dir: Union[os.PathLike, str], file_name: str) -> dict:
    file_path = _json_file_path(source_dir, file_name)

    if file_path:
        with open(file_path, "r") as f:
            return simplejson.load(f)
    else:
        raise FileNotFoundError(f"'{file_name}' file not found in '{source_dir}'")


def load_metadata(source_dir: Union[os.PathLike, str]) -> dict:
    """
    Load the given metadata.json which was saved during the ``serializer.dump``
    will return the loaded metadata as a dict, or empty dict if no file was found

    Parameters
    ----------
    source_dir
        Directory of the saved model, As with serializer.load(source_dir) this
        source_dir can be the top level, or the first dir into the serialized model.

    Returns
    -------

    Raises
    ------
    FileNotFoundError
        If a 'metadata.json' file isn't found in or above the supplied ``source_dir``
    """
    return _load_json_file(source_dir, "metadata.json")


def load_info(source_dir: Union[os.PathLike, str]) -> dict:
    # TODO better docstring
    return _load_json_file(source_dir, "info.json")


def load(source_dir: Union[os.PathLike, str]) -> Any:
    """
    Load an object from a directory, saved by
    ``gordo.serializer.pipeline_serializer.dump``

    This take a directory, which is either top-level, meaning it contains
    a sub directory in the naming scheme: "n_step=<int>-class=<path.to.Class>"
    or the aforementioned naming scheme directory directly. Will return that
    unsterilized object.


    Parameters
    ----------
    source_dir
        Location of the top level dir the pipeline was saved

    Returns
    -------
    """
    # This source dir should have a single pipeline entry directory.
    # may have been passed a top level dir, containing such an entry:
    with open(os.path.join(source_dir, "model.pkl"), "rb") as f:
        return pickle.load(f)


def dump(
    obj: object,
    dest_dir: Union[os.PathLike, str],
    metadata: dict = None,
    info: Optional[dict] = None,
):
    """
    Serialize an object into a directory, the object must be pickle-able.

    Parameters
    ----------
    obj
        The object to dump. Must be pickle-able.
    dest_dir
        The directory to which to save the model metadata: dict - any additional
        metadata to be saved alongside this model if it exists, will be returned
        from the corresponding "load" function
    metadata
        with the model, and loaded again by :func:`load_metadata`.
    info
        Current revision info. For now, only used for storing "checksum"

    Returns
    -------

    Example
    -------

    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.decomposition import PCA
    >>> from gordo.machine.model.models import KerasAutoEncoder
    >>> from gordo import serializer
    >>> from tempfile import TemporaryDirectory
    >>> pipe = Pipeline([
    ...     ('pca', PCA(3)),
    ...     ('model', KerasAutoEncoder(kind='feedforward_hourglass'))])
    >>> with TemporaryDirectory() as tmp:
    ...     serializer.dump(obj=pipe, dest_dir=tmp)
    ...     pipe_clone = serializer.load(source_dir=tmp)
    """
    with open(os.path.join(dest_dir, "model.pkl"), "wb") as m:
        pickle.dump(obj, m)
    if info is not None:
        with open(os.path.join(dest_dir, "info.json"), "w") as f:
            simplejson.dump(info, f, default=str)
    if metadata is not None:
        with open(os.path.join(dest_dir, "metadata.json"), "w") as f:
            simplejson.dump(metadata, f, default=str)
