# -*- coding: utf-8 -*-
import datetime
from distutils.dir_util import copy_tree
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Union, Optional, Dict, Any, Tuple

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator
from sklearn.metrics import (
    explained_variance_score,
    make_scorer,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)
from sklearn.model_selection import cross_validate, TimeSeriesSplit
from sklearn.pipeline import Pipeline

from gordo_components.util import disk_registry
from gordo_components import serializer, __version__, MAJOR_VERSION, MINOR_VERSION
from gordo_components.dataset.dataset import _get_dataset
from gordo_components.dataset.base import GordoBaseDataset
from gordo_components.model.base import GordoBase
from gordo_components.model.utils import metric_wrapper

logger = logging.getLogger(__name__)


def build_model(
    name: str,
    model_config: dict,
    data_config: Union[GordoBaseDataset, dict],
    metadata: dict,
    evaluation_config: dict = {"cv_mode": "full_build"},
) -> Tuple[Union[BaseEstimator, None], dict]:
    """
    Build a model and serialize to a directory for later serving.

    Parameters
    ----------
    name: str
        Name of model to be built
    model_config: dict
        Mapping of Model to initialize and any additional kwargs which are to be used in it's initialization.
        Example::

          {'type': 'KerasAutoEncoder',
           'kind': 'feedforward_hourglass'}

    data_config: dict
        Mapping of the Dataset to initialize, following the same logic as model_config.
    metadata: dict
        Mapping of arbitrary metadata data.
    evaluation_config: dict
        Dict of parameters which are exposed to build_model.
            - cv_mode: str
                String which enables three different modes, represented as a key value in evaluation_config:
                * cross_val_only: Only perform cross validation
                * build_only: Skip cross validation and only build the model
                * full_build: Cross validation and full build of the model, default value
                Example::

                    {"cv_mode": "cross_val_only"}


    Returns
    -------
        Tuple[Optional[sklearn.base.BaseEstimator], dict]
    """
    # Get the dataset from config
    logger.debug(f"Initializing Dataset with config {data_config}")

    dataset = (
        data_config
        if isinstance(data_config, GordoBaseDataset)
        else _get_dataset(data_config)
    )

    logger.debug("Fetching training data")
    start = time.time()

    X, y = dataset.get_data()

    time_elapsed_data = time.time() - start

    # Get the model and dataset
    logger.debug(f"Initializing Model with config: {model_config}")
    model = serializer.pipeline_from_definition(model_config)

    cv_duration_sec = None

    if evaluation_config["cv_mode"].lower() in ("cross_val_only", "full_build"):
        metrics_list = [
            explained_variance_score,
            r2_score,
            mean_squared_error,
            mean_absolute_error,
        ]
        # Cross validate
        logger.debug("Starting cross validation")
        start = time.time()
        scores: Dict[str, Any] = dict()
        if hasattr(model, "predict"):

            metrics_dict = get_metrics_dict(metrics_list, y)

            cv = cross_validate(
                model,
                X,
                y,
                scoring=metrics_dict,
                return_estimator=True,
                cv=TimeSeriesSplit(n_splits=3),
            )
            for metric, test_metric in map(lambda k: (k, f"test_{k}"), metrics_dict):
                val = {
                    "fold-mean": cv[test_metric].mean(),
                    "fold-std": cv[test_metric].std(),
                    "fold-max": cv[test_metric].max(),
                    "fold-min": cv[test_metric].min(),
                }
                val.update(
                    {
                        f"fold-{i + 1}": raw_value
                        for i, raw_value in enumerate(cv[test_metric].tolist())
                    }
                )
                scores.update({metric: val})

        else:
            logger.debug("Unable to score model, has no attribute 'predict'.")
            scores = dict()

        cv_duration_sec = time.time() - start

        # If cross_val_only, return the cv_scores and empty model.
        if evaluation_config["cv_mode"] == "cross_val_only":
            metadata["model"] = {
                "cross-validation": {
                    "cv-duration-sec": cv_duration_sec,
                    "scores": scores,
                }
            }
            return None, metadata
    else:
        # Setting cv scores to zero when not used.
        scores = dict()
    # Train
    logger.debug("Starting to train model.")
    start = time.time()
    model.fit(X, y)
    time_elapsed_model = time.time() - start

    metadata = {"user-defined": metadata}
    metadata["name"] = name
    metadata["dataset"] = dataset.get_metadata()
    utc_dt = datetime.datetime.now(datetime.timezone.utc)
    metadata["model"] = {
        "model-offset": _determine_offset(model, X),
        "model-creation-date": str(utc_dt.astimezone()),
        "model-builder-version": __version__,
        "model-config": model_config,
        "data-query-duration-sec": time_elapsed_data,
        "model-training-duration-sec": time_elapsed_model,
        "cross-validation": {"cv-duration-sec": cv_duration_sec, "scores": scores},
    }

    metadata["model"].update(_get_metadata(model))
    return model, metadata


def get_metrics_dict(metrics_list: list, y: pd.DataFrame) -> dict:

    """
    Given a list of metrics that accept a true_y and pred_y as inputs this returns
    a dictionary with keys in the form '{score}-{tag_name}' for each given target tag
    and '{score}' for the average score across all target tags and folds,
    and values being the callable make_scorer(metric_wrapper(score)).
    Note: score in {score}-{tag_name} is a sklearn's score function name with '_' replaced by '-' and tag_name
    corresponds to given target tag name with ' ' replaced by '-'.

    Parameters
    ----------
    metrics_list: list
        List of sklearn score functions
    y: pd.DataFrame
        Target data


    Returns
    -------
        dict
    """

    def _score_factory(metric=r2_score, index=0):
        def _score_per_tag(y_true, y_pred):
            # This function extracts the score for each given target_tag to
            # use as scoring argument in sklearn cross_validate, as the scoring
            # must return a single value.
            if hasattr(y_true, "values"):
                y_true = y_true.values
            if hasattr(y_pred, "values"):
                y_pred = y_pred.values

            return metric(y_true[:, index], y_pred[:, index])

        return _score_per_tag

    metrics_dict = {}
    for metric in metrics_list:
        for index, col in enumerate(y.columns):
            metric_str = metric.__name__.replace("_", "-")
            metrics_dict.update(
                {
                    metric_str
                    + f'-{col.replace(" ", "-")}': make_scorer(
                        metric_wrapper(_score_factory(metric=metric, index=index))
                    )
                }
            )

        metrics_dict.update({metric_str: make_scorer(metric_wrapper(metric))})
    return metrics_dict


def _determine_offset(model: BaseEstimator, X: Union[np.ndarray, pd.DataFrame]) -> int:
    """
    Determine the model's offset. How much does the output of the model differ
    from its input?

    Parameters
    ----------
    model: sklearn.base.BaseEstimator
        Trained model with either ``predict`` or ``transform`` method, preference
        given to ``predict``.
    X: Union[np.ndarray, pd.DataFrame]
        Data to pass to the model's ``predict`` or ``transform`` method.

    Returns
    -------
    int
        The difference between X and the model's output lengths.
    """
    out = model.predict(X) if hasattr(model, "predict") else model.transform(X)
    return len(X) - len(out)


def _save_model_for_workflow(
    model: BaseEstimator, metadata: dict, output_dir: Union[os.PathLike, str]
):
    """
    Save a model according to the expected Argo workflow procedure.

    Parameters
    ----------
    model: BaseEstimator
        The model to save to the directory with gordo serializer.
    metadata: dict
        Various mappings of metadata to save alongside model.
    output_dir: Union[os.PathLike, str]
        The directory where to save the model, will create directories if needed.

    Returns
    -------
    Union[os.PathLike, str]
        Path to the saved model
    """
    os.makedirs(output_dir, exist_ok=True)  # Ok if some dirs exist
    serializer.dump(model, output_dir, metadata=metadata)
    return output_dir


def _get_metadata(model: BaseEstimator, metadata: dict = dict()) -> dict:
    """
    Recursively check for :class:`gordo_components.model.base.GordoBase` in a
    given ``model``. If such a model exists buried inside of a
    :class:`sklearn.pipeline.Pipeline` which is then part of another
    :class:`sklearn.base.BaseEstimator`, this function will return its metadata.

    Parameters
    ----------
    model: BaseEstimator
    metadata: dict
        Any initial starting metadata, but is mainly meant to be used during
        the recursive calls to accumulate any multiple :class:`gordo_components.model.base.GordoBase`
        models found in this model

    Notes
    -----
    If there is a ``GordoBase`` model inside of a ``Pipeline`` which is not the final
    step, this function will not find it.

    Returns
    -------
    dict
        Dictionary representing accumulated calls to :meth:`gordo_components.model.base.GordoBase.get_metadata`
    """
    metadata = metadata.copy()

    # If it's a Pipeline, only need to get the last step, which potentially has metadata
    if isinstance(model, Pipeline):
        final_step = model.steps[-1][1]
        metadata.update(_get_metadata(final_step))
        return metadata

    # GordoBase is simple, having a .get_metadata()
    if isinstance(model, GordoBase):
        metadata.update(model.get_metadata())

    # Continue to look at object values in case, we decided to have a GordoBase
    # which also had a GordoBase as a parameter/attribute, but will satisfy BaseEstimators
    # which can take a GordoBase model as a parameter, which will then have metadata to get
    for val in model.__dict__.values():
        if isinstance(val, Pipeline):
            metadata.update(_get_metadata(val.steps[-1][1]))
        elif isinstance(val, GordoBase) or isinstance(val, BaseEstimator):
            metadata.update(_get_metadata(val))
    return metadata


def calculate_model_key(
    name: str,
    model_config: dict,
    data_config: dict,
    evaluation_config: dict,
    metadata: Optional[dict] = None,
) -> str:
    """
    Calculates a hash-key from a model and data-config.

    Notes
    -----
    Ignores the data_provider key since this is an complicated object.

    Parameters
    ----------
    name: str
        Name of the model
    model_config: dict
        Config for the model. See
        :func:`gordo_components.builder.build_model.build_model`.
    data_config: dict
        Config for the data-configuration. See
        :func:`gordo_components.builder.build_model.build_model`.
    evaluation_config: dict
        Config for the evaluation-configuration. See
        :func:`gordo_components.builder.build_moodel.build_model`.
    metadata: Optional[dict] = None
        Metadata for the models. See
        :func:`gordo_components.builder.build_model.build_model`.

    Returns
    -------
    str:
        A 512 byte hex value as a string based on the content of the parameters.

    Examples
    -------
    >>> len(calculate_model_key(name="My-model", model_config={"model": "something"},
    ... data_config={"tag_list": ["tag1", "tag 2"]}, evaluation_config={"cv_mode": "full_build"} ))
    128
    """
    if metadata is None:
        metadata = {}
    # TODO Fix this when we get a good way of passing data_provider in the yaml/json
    if "data_provider" in data_config:
        logger.warning(
            "data_provider key found in data_config, ignoring it when creating hash"
        )
        data_config = dict(data_config)
        del data_config["data_provider"]

    # Sets a lot of the parameters to json.dumps explicitly to ensure that we get
    # consistent hash-values even if json.dumps changes their default values (and as such might
    # generate different json which again gives different hash)
    json_rep = json.dumps(
        {
            "name": name,
            "model_config": model_config,
            "data_config": data_config,
            "user-defined": metadata,
            "evaluation_config": evaluation_config,
            "gordo-major-version": MAJOR_VERSION,
            "gordo-minor-version": MINOR_VERSION,
        },
        sort_keys=True,
        default=str,
        skipkeys=False,
        ensure_ascii=True,
        check_circular=True,
        allow_nan=True,
        cls=None,
        indent=None,
        separators=None,
    )
    logger.debug(f"Calculating model hash key for model: {json_rep}")
    return hashlib.sha3_512(json_rep.encode("ascii")).hexdigest()


def check_cache(model_register_dir: Union[os.PathLike, str], cache_key: str):
    """
    Checks if a model is cached, and returns its path if it exists.

    Parameters
    ----------
    model_register_dir: [os.PathLike, None]
        The register dir where the model lies.
    cache_key: str
        A 512 byte hex value as a string based on the content of the parameters.

     Returns
    -------
    Union[os.PathLike, None]:
        The path to the cached model, or None if it does not exist.
    """
    existing_model_location = disk_registry.get_value(model_register_dir, cache_key)

    # Check that the model is actually there
    if existing_model_location and Path(existing_model_location).exists():
        logger.debug(
            f"Found existing model at path {existing_model_location}, returning it"
        )
        return existing_model_location
    elif existing_model_location:
        logger.warning(
            f"Found that the model-path {existing_model_location} stored in the "
            f"registry did not exist."
        )
        return None
    else:
        logger.info(
            f"Did not find the model with key {cache_key} in the register at "
            f"{model_register_dir}."
        )
        return None


def provide_saved_model(
    name: str,
    model_config: dict,
    data_config: dict,
    metadata: dict,
    output_dir: Union[os.PathLike, str],
    model_register_dir: Union[os.PathLike, str] = None,
    replace_cache=False,
    evaluation_config: dict = {"cv_mode": "full_build"},
) -> Union[os.PathLike, str]:
    """
    Ensures that the desired model exists on disk in `output_dir`, and returns the path
    to it. If `output_dir` exists we assume the model is there (no validation), and
    return that path.


    Builds the model if needed, or finds it among already existing models if
    ``model_register_dir`` is non-None, and we find the model there. If
    `model_register_dir` is set we will also store the model-location of the generated
    model there for future use. Think about it as a cache that is never emptied.

    Parameters
    ----------
    name: str
        Name of the model to be built
    model_config: dict
        Config for the model. See
        :func:`gordo_components.builder.build_model.build_model`.
    data_config: dict
        Config for the data-configuration. See
        :func:`gordo_components.builder.build_model.build_model`.
    metadata: dict
        Extra metadata to be added to the built models if it is built. See
        :func:`gordo_components.builder.build_model.build_model`.
    output_dir: Union[os.PathLike, str]
        A path to where the model will be deposited if it is built.
    model_register_dir:
        A path to a register, see `gordo_components.util.disk_registry`. If this is None
        then always build the model, otherwise try to resolve the model from the
        registry.
    replace_cache: bool
        Forces a rebuild of the model, and replaces the entry in the cache with the new
        model.
    evaluation_config: dict
        Config for the evaluation. See
        :func:`gordo_components.builder.build_model.build_model`.

    Returns
    -------
    Union[os.PathLike, str]:
        Path to the model
    """
    cache_key = calculate_model_key(
        name, model_config, data_config, evaluation_config, metadata=metadata
    )
    if model_register_dir:
        logger.info(
            f"Model caching activated, attempting to read model-location with key "
            f"{cache_key} from register {model_register_dir}"
        )
        if replace_cache:
            logger.info("replace_cache activated, deleting any existing cache entry")
            disk_registry.delete_value(model_register_dir, cache_key)
        else:
            cached_model_location = check_cache(model_register_dir, cache_key)
            if cached_model_location:
                logger.info(
                    f"Found model in cache, copying from {cached_model_location} to "
                    f"new location {output_dir} "
                )
                if cached_model_location == output_dir:
                    return output_dir
                else:
                    try:
                        # Why not shutil.copytree? Because in python <3.7 it causes
                        # errors on Azure NFS, see:
                        # - https://bugs.python.org/issue24564
                        # - https://stackoverflow.com/questions/51616058/shutil-copystat-fails-inside-docker-on-azure/51635427#51635427
                        copy_tree(
                            str(cached_model_location),
                            str(output_dir),
                            preserve_mode=0,
                            preserve_times=0,
                        )
                    except FileExistsError:
                        logger.warning(
                            f"Found that output directory {output_dir} "
                            f"already exists, assuming model is "
                            f"already located there"
                        )
                    return output_dir

    model, metadata = build_model(
        name=name,
        model_config=model_config,
        data_config=data_config,
        metadata=metadata,
        evaluation_config=evaluation_config,
    )
    model_location = _save_model_for_workflow(
        model=model, metadata=metadata, output_dir=output_dir
    )
    logger.info(f"Successfully built model, and deposited at {model_location}")
    if model_register_dir:
        logger.info(f"Writing model-location to model registry")
        disk_registry.write_key(model_register_dir, cache_key, model_location)
    return model_location
