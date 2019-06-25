# -*- coding: utf-8 -*-
import datetime
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Union, Optional, Dict, Any

from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.pipeline import Pipeline

from gordo_components.util import disk_registry
from gordo_components import serializer, __version__, MAJOR_VERSION, MINOR_VERSION
from gordo_components.dataset.dataset import _get_dataset
from gordo_components.dataset.base import GordoBaseDataset
from gordo_components.model.base import GordoBase

logger = logging.getLogger(__name__)


def build_model(
    name: str,
    model_config: dict,
    data_config: Union[GordoBaseDataset, dict],
    metadata: dict,
):
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

    Returns
    -------
        Tuple[sklearn.base.BaseEstimator, dict]
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
    end = time.time()
    time_elapsed_data = end - start

    # Get the model and dataset
    logger.debug(f"Initializing Model with config: {model_config}")
    model = serializer.pipeline_from_definition(model_config)

    # Cross validate
    logger.debug(f"Starting to do cross validation")
    start = time.time()

    scores: Dict[str, Any]
    if hasattr(model, "score"):
        cv_scores = cross_val_score(model, X, y, cv=TimeSeriesSplit(n_splits=3))
        scores = {
            "explained-variance": {
                "mean": cv_scores.mean(),
                "std": cv_scores.std(),
                "max": cv_scores.max(),
                "min": cv_scores.min(),
                "raw-scores": cv_scores.tolist(),
            }
        }
    else:
        logger.debug("Unable to score model, has no attribute 'score'.")
        scores = dict()

    cv_duration_sec = time.time() - start

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
        "model-creation-date": str(utc_dt.astimezone()),
        "model-builder-version": __version__,
        "model-config": model_config,
        "data-query-duration-sec": time_elapsed_data,
        "model-training-duration-sec": time_elapsed_model,
        "cross-validation": {"cv-duration-sec": cv_duration_sec, "scores": scores},
    }

    gordobase_final_step = _get_final_gordo_base_step(model)
    if gordobase_final_step:
        metadata["model"].update(gordobase_final_step.get_metadata())

    return model, metadata


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


def _get_final_gordo_base_step(model: BaseEstimator):
    """
    Get the final GordoBase step in a (potential) Pipeline, if it exists.
    Parameters
    ----------
    model: BaseEstimator
        The input model or Pipeline to investigate. If a Pipeline is given, look for
        the last step in (the possibly nested) Pipeline.

    Returns
    -------
    GordoBase
        The final GordoBase object in the pipeline, or None if not found.

    """
    if isinstance(model, GordoBase):
        return model

    elif isinstance(model, Pipeline):
        last_step_tuple = model.steps[-1]  # Get the last step tuple
        estimator = last_step_tuple[1]  # The actual step is the second element
        return _get_final_gordo_base_step(estimator)

    else:
        return None


def calculate_model_key(
    name: str, model_config: dict, data_config: dict, metadata: Optional[dict] = None
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
    ... data_config={"tag_list": ["tag1", "tag 2"]} ))
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


def provide_saved_model(
    name: str,
    model_config: dict,
    data_config: dict,
    metadata: dict,
    output_dir: Union[os.PathLike, str],
    model_register_dir: Union[os.PathLike, str] = None,
    replace_cache=False,
) -> Union[os.PathLike, str]:
    """
    Ensures that the desired model exists on disk, and returns the path to it.

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

    Returns
    -------
    Union[os.PathLike, str]:
        Path to the model
    """
    cache_key = calculate_model_key(name, model_config, data_config, metadata=metadata)
    if model_register_dir:
        logger.info(
            f"Model caching activated, attempting to read model-location with key "
            f"{cache_key} from register {model_register_dir}"
        )
        if replace_cache:
            logger.info("replace_cache activated, deleting any existing cache entry")
            cache_key = calculate_model_key(
                name, model_config, data_config, metadata=metadata
            )
            disk_registry.delete_value(model_register_dir, cache_key)

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
        else:
            logger.info(
                f"Did not find the model with key {cache_key} in the register at "
                f"{model_register_dir}."
            )
    model, metadata = build_model(
        name=name, model_config=model_config, data_config=data_config, metadata=metadata
    )
    model_location = _save_model_for_workflow(
        model=model, metadata=metadata, output_dir=output_dir
    )
    logger.info(f"Successfully built model, and deposited at {model_location}")
    if model_register_dir:
        logger.info(f"Writing model-location to model registry")
        disk_registry.write_key(model_register_dir, cache_key, model_location)
    return model_location
