# -*- coding: utf-8 -*-

import bz2
import glob
import io
import simplejson
import logging
import os
import pydoc
import re
import pickle
import tarfile
import tempfile

from os import path
from typing import Tuple, Union, Dict, Any, IO  # pragma: no flakes

from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.base import TransformerMixin, BaseEstimator  # noqa

from gordo_components.model.base import GordoBase

logger = logging.getLogger(__name__)

N_STEP_REGEX = re.compile(r".*n_step=([0-9]+)")
CLASS_REGEX = re.compile(r".*class=(.*$)")


def dumps(model: Union[Pipeline, GordoBase]) -> bytes:
    """
    Dump a model into a bytes representation suitable for loading from
    ``gordo_components.serializer.loads``

    Parameters
    ----------
    model: Union[Pipeline, GordoBase]
        A gordo model/pipeline

    Returns
    -------
    bytes
        Serialized model which supports loading via ``serializer.loads()``

    Example
    -------
    >>> from gordo_components.model.models import KerasAutoEncoder
    >>> from gordo_components import serializer
    >>>
    >>> model = KerasAutoEncoder('feedforward_symmetric')
    >>> serialized = serializer.dumps(model)
    >>> assert isinstance(serialized, bytes)
    >>>
    >>> model_clone = serializer.loads(serialized)
    >>> assert isinstance(model_clone, KerasAutoEncoder)
    """
    with tempfile.TemporaryDirectory() as tmp:
        dump(model, tmp)
        tarbuff = io.BytesIO()
        with tarfile.open(fileobj=tarbuff, mode="w:gz") as archive:
            archive.add(tmp, recursive=True, arcname="serialized_gordo_model")
        tarbuff.seek(0)
        return tarbuff.read()


def loads(bytes_object: bytes) -> GordoBase:
    """
    Load a GordoBase model from bytes dumped from ``gordo_components.serializer.dumps``

    Parameters
    ----------
    bytes_object: bytes
        Bytes to be loaded, should be the result of `serializer.dumps(model)`

    Returns
    -------
    Union[GordoBase, Pipeline, BaseEstimator]
        Custom gordo model, scikit learn pipeline or other scikit learn like object.
    """
    with tempfile.TemporaryDirectory() as tmp:

        tarbuff = io.BytesIO(bytes_object)
        tarbuff.seek(0)

        with tarfile.open(fileobj=tarbuff, mode="r:gz") as archive:
            archive.extractall(tmp)
        return load(os.path.join(tmp, "serialized_gordo_model"))


def load_metadata(source_dir: str) -> dict:
    """
    Load the given metadata.json which was saved during the ``serializer.dump``
    will return the loaded metadata as a dict, or empty dict if no file was found

    Parameters
    ----------
    source_dir: str
        Directory of the saved model, As with serializer.load(source_dir) this
        source_dir can be the top level, or the first dir into the serialized model.

    Returns
    -------
    dict
    """
    # Since this function can take the top level dir, or a dir directly
    # into the first step of the pipeline, we need to check both for metadata
    for possible_path in [
        path.join(source_dir, "metadata.json"),
        path.join(source_dir, "..", "metadata.json"),
    ]:
        if path.isfile(possible_path):
            with open(possible_path, "r") as f:
                return simplejson.load(f)
    logger.warning(
        f'Metadata file in source dir: "{source_dir}" not found'
        f" in or up one directory."
    )
    return dict()


def load(source_dir: str) -> Any:
    """
    Load an object from a directory, saved by
    ``gordo_components.serializer.pipeline_serializer.dump``

    This take a directory, which is either top-level, meaning it contains
    a sub directory in the naming scheme: "n_step=<int>-class=<path.to.Class>"
    or the aforementioned naming scheme directory directly. Will return that
    unsterilized object.


    Parameters
    ----------
    source_dir: str
        Location of the top level dir the pipeline was saved

    Returns
    -------
    Union[GordoBase, Pipeline, BaseEstimator]
    """
    # This source dir should have a single pipeline entry directory.
    # may have been passed a top level dir, containing such an entry:
    if not source_dir.startswith("n_step"):
        dirs = [d for d in os.listdir(source_dir) if "n_step=" in d]
        if len(dirs) != 1:
            raise ValueError(
                f"Found multiple object entries to load from, "
                f"should pass a directory to pipeline directly or "
                f"a directory containing a single object entry."
                f"Possible objects found: {dirs}"
            )
        else:
            source_dir = path.join(source_dir, dirs[0])

    # Load step always returns a tuple of (str, object), index to object
    return _load_step(source_dir)[1]


def _parse_dir_name(source_dir: str) -> Tuple[int, str]:
    """
    Parses the required params from a directory name for loading
    Expected name format ``n_step=<int>-class=<path.to.class.Model>``
    """
    match = N_STEP_REGEX.search(source_dir)
    if match is None:
        raise ValueError(
            f'Source dir not valid, expected "n_step=" in '
            f"directory but instead got: {source_dir}"
        )
    else:
        n_step = int(match.groups()[0])  # type: int

    match = CLASS_REGEX.search(source_dir)
    if match is None:
        raise ValueError(
            f'Source dir not valid, expected "class=" in directory '
            f"but instead got: {source_dir}"
        )
    else:
        class_path = match.groups()[0]  # type: str
    return n_step, class_path


def _load_step(source_dir: str) -> Tuple[str, object]:
    """
    Load a single step from a source directory

    Parameters
    ----------
        source_dir: str - directory in format "n_step=<int>-class=<path.to.class.Model>"

    Returns
    -------
        Tuple[str, object]
    """
    n_step, class_path = _parse_dir_name(source_dir)
    StepClass = pydoc.locate(
        class_path
    )  # type: Union[FeatureUnion, Pipeline, BaseEstimator]
    if StepClass is None:
        logger.warning(
            f'Specified a class path of "{class_path}" but it does '
            f"not exist. Will attempt to unpickle it from file in "
            f"source directory: {source_dir}."
        )
    step_name = f"step={str(n_step).zfill(3)}"
    params = dict()  # type: Dict[str, Any]

    # If this is a FeatureUnion, we also have a `params.json` for it
    if StepClass == FeatureUnion:
        with open(os.path.join(source_dir, "params.json"), "r") as p:
            params = simplejson.load(p)

    # Pipelines and FeatureUnions have sub steps which need to be loaded
    if any(StepClass == Obj for Obj in (Pipeline, FeatureUnion)):

        # Load the sub_dirs to load into the Pipeline/FeatureUnion in order
        sub_dirs_to_load = sorted(
            [
                sub_dir
                for sub_dir in os.listdir(source_dir)
                if path.isdir(path.join(source_dir, sub_dir))
            ],
            key=lambda d: _parse_dir_name(d)[0],
        )
        steps = [
            _load_step(path.join(source_dir, sub_dir)) for sub_dir in sub_dirs_to_load
        ]
        return step_name, StepClass(steps, **params)

    # May model implementing load_from_dir method, from GordoBase
    elif hasattr(StepClass, "load_from_dir"):
        return step_name, StepClass.load_from_dir(source_dir)

    # Otherwise we have a normal Scikit-Learn transformer
    else:
        # Find the name of this file in the directory, should only be one
        file = glob.glob(path.join(source_dir, "*.pkl.gz"))
        if len(file) != 1:
            raise ValueError(
                f"Expected a single file in what is expected to be "
                f"a single object directory, found {len(file)} "
                f"in directory: {source_dir}"
            )
        with bz2.open(path.join(source_dir, file[0]), "rb") as f:  # type: IO[bytes]
            model = pickle.load(f)

        # This model may have been an estimator which took a GordoBase as a parameter
        # and would have had that parameter/model dumped to a seperate directory and
        # replaced the attribute with the location of that model.
        for attr_name, attr_value in model.__dict__.items():
            if isinstance(attr_value, dict) and all(
                k in attr_value for k in ("class_path", "load_dir")
            ):
                class_path, load_dir = attr_value["class_path"], attr_value["load_dir"]
                if class_path is not None:
                    GordoModel: GordoBase = pydoc.locate(class_path)  # type: ignore
                    attr_model = GordoModel.load_from_dir(load_dir)
                else:
                    # Otherwise it was another attr which required dumping
                    attr_model = load(load_dir)
                setattr(model, attr_name, attr_model)
        return step_name, model


def dump(obj: object, dest_dir: Union[os.PathLike, str], metadata: dict = None):
    """
    Serialize an object into a directory

    The object must either be picklable or implement BOTH a ``GordoBase.save_to_dir`` AND
    ``GordoBase.load_from_dir`` methods. This object can hold multiple objects, specifically
    it can be a sklearn.pipeline.[FeatureUnion, Pipeline] object, in such a case
    it's sub transformers (steps/transformer_list) will be serialized recursively.

    Parameters
    ----------
    obj
        The object to dump. Must be picklable or implement
        a ``save_to_dir`` AND ``load_from_dir`` method.
    dest_dir: Union[os.PathLike, str]
        The directory to which to save the model metadata: dict - any additional
        metadata to be saved alongside this model if it exists, will be returned
        from the corresponding "load" function
    metadata: Optional dict of metadata which will be serialized to a file together
        with the model, and loaded again by :func:`load_metadata`.

    Returns
    -------
    None

    Example
    -------

    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.decomposition import PCA
    >>> from gordo_components.model.models import KerasAutoEncoder
    >>> from gordo_components import serializer
    >>> from tempfile import TemporaryDirectory
    >>> pipe = Pipeline([
    ...     # PCA is picklable
    ...     ('pca', PCA(3)),
    ...     # KerasAutoEncoder implements both `save_to_dir` and `load_from_dir`
    ...     ('model', KerasAutoEncoder(kind='feedforward_hourglass'))])
    >>> with TemporaryDirectory() as tmp:
    ...     serializer.dump(obj=pipe, dest_dir=tmp)
    ...     pipe_clone = serializer.load(source_dir=tmp)
    """
    _dump_step(step=("obj", obj), n_step=0, dest_dir=dest_dir)
    if metadata is not None:
        with open(os.path.join(dest_dir, "metadata.json"), "w") as f:
            simplejson.dump(metadata, f, default=str)


def _dump_step(
    step: Tuple[str, Union[GordoBase, TransformerMixin]],
    dest_dir: Union[os.PathLike, str],
    n_step: int = 0,
):
    """
    Accepts any Scikit-Learn transformer and dumps it into a directory
    recoverable by gordo_components.serializer.pipeline_serializer.load

    Parameters
    ----------
    step
        The step to dump
    dest_dir
        The path to the top level directory to start the potentially recursive saving of steps.
    n_step
        The order of this step in the pipeline, default to 0

    Returns
    -------
    None
        Creates a new directory at the `dest_dir` in the format:
        `n_step=000-class=<full.path.to.Object` with any required files for recovery stored there.
    """
    step_name, step_transformer = step
    step_import_str = (
        f"{step_transformer.__module__}.{step_transformer.__class__.__name__}"
    )
    sub_dir = os.path.join(
        dest_dir, f"n_step={str(n_step).zfill(3)}-class={step_import_str}"
    )

    os.makedirs(sub_dir, exist_ok=True)

    if any(isinstance(step_transformer, Obj) for Obj in [FeatureUnion, Pipeline]):
        steps_attr = (
            "transformer_list"
            if isinstance(step_transformer, FeatureUnion)
            else "steps"
        )
        for i, step in enumerate(getattr(step_transformer, steps_attr)):
            _dump_step(step=step, n_step=i, dest_dir=sub_dir)

        # If this is a feature Union, we want to save `n_jobs` & `transformer_weights`
        if isinstance(step_transformer, FeatureUnion):
            params = {
                "n_jobs": getattr(step_transformer, "n_jobs"),
                "transformer_weights": getattr(step_transformer, "transformer_weights"),
            }
            with open(os.path.join(sub_dir, "params.json"), "w") as f:
                simplejson.dump(params, f)
    else:

        if isinstance(step_transformer, GordoBase):
            if not hasattr(step_transformer, "load_from_dir"):
                raise AttributeError(
                    f'The object in this step implements a "save_to_dir" but '
                    f'not "load_from_dir" it will be un-recoverable!'
                )
            logger.info(f"Saving model to sub_dir: {sub_dir}")
            step_transformer.save_to_dir(os.path.join(sub_dir))

        # This may be a transformer which has a GordoBase as a parameter
        else:

            # We have to ensure this is a pickle-able transformer, in that it doesn't
            # have any GordoBase models hiding as a parameter to it.
            attrs_dir = os.path.join(sub_dir, "_gordo_base_attributes")
            for attr_name, attr_value in step_transformer.__dict__.items():
                if isinstance(attr_value, Pipeline):
                    attr_serialization_dir = os.path.join(attrs_dir, attr_name)
                    new_attr_val = {
                        "class_path": None,
                        "load_dir": attr_serialization_dir,
                    }
                    dump(attr_value, attr_serialization_dir)
                    setattr(step_transformer, attr_name, new_attr_val)

                elif isinstance(attr_value, GordoBase):
                    attr_serialization_dir = os.path.join(attrs_dir, attr_name)
                    os.makedirs(attr_serialization_dir, exist_ok=True)

                    attr_value.save_to_dir(attr_serialization_dir)
                    new_attr_val = {
                        "class_path": f"{attr_value.__module__}.{attr_value.__class__.__name__}",
                        "load_dir": attr_serialization_dir,
                    }
                    setattr(step_transformer, attr_name, new_attr_val)

            with bz2.open(
                os.path.join(sub_dir, f"{step_name}.pkl.gz"), "wb"
            ) as s:  # type: IO[bytes]
                pickle.dump(step_transformer, s)
