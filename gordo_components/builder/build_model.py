# -*- coding: utf-8 -*-

import datetime
from distutils.dir_util import copy_tree
import hashlib
import json
import pydoc
import logging
import os
import time
import random
from pathlib import Path
from typing import Union, Optional, Dict, Any, Tuple, List, Callable

import pandas as pd
import numpy as np
import tensorflow as tf

import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import metrics
from sklearn.model_selection import cross_validate, TimeSeriesSplit
from sklearn.pipeline import Pipeline

from gordo_components.util import disk_registry
from gordo_components import serializer, __version__, MAJOR_VERSION, MINOR_VERSION
from gordo_components.dataset.dataset import _get_dataset
from gordo_components.dataset.base import GordoBaseDataset
from gordo_components.model.base import GordoBase
from gordo_components.model.utils import metric_wrapper
from gordo_components.workflow.config_elements.normalized_config import NormalizedConfig

logger = logging.getLogger(__name__)


class ModelBuilder:
    def __init__(
        self,
        name: str,
        model_config: dict,
        data_config: dict,
        metadata: dict = dict(),
        evaluation_config: dict = {
            "cv_mode": "full_build",
            "scoring_scaler": "sklearn.preprocessing.RobustScaler",
        },
    ):
        """
        Use the raw data from Gordo config file keys: name, model, dataset,
        metadata, and evalution to build the final ML model.

        Parameters
        ----------
        name: str
            Name of model to be built
        model_config: dict
            Mapping of Model to initialize and any additional kwargs which are
            to be used in it's initialization.
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
                    String which enables three different modes, represented as a
                    key value in evaluation_config:
                    * cross_val_only: Only perform cross validation
                    * build_only: Skip cross validation and only build the model
                    * full_build: Cross validation and full build of the model, default value
                - scoring_scaler: Optional[str]
                    Optionally a string which gives the path to a scaler, which will be
                    applied on `Y` and `Y-hat` before scoring. It will not be used for
                    training or any other part of the model-building, only for the scoring.
                    This is usefull if one has multi-dimensional output, where the different
                    dimensions have different scale, and one wants to use a scorer which is
                    sensitive to this (e.g. MSE).
                    If None then no scaling will take place.

                    Example::

                        {"cv_mode": "cross_val_only",
                        "scoring_scaler": "sklearn.preprocessing.RobustScaler"}

        Example
        -------
        >>> from gordo_components.dataset.sensor_tag import SensorTag
        >>> builder = ModelBuilder(
        ...     name="special-model-name",
        ...     model_config={"sklearn.decomposition.pca.PCA": {"svd_solver": "auto"}},
        ...     data_config={
        ...         "type": "RandomDataset",
        ...         "train_start_date": "2017-12-25 06:00:00Z",
        ...         "train_end_date": "2017-12-30 06:00:00Z",
        ...         "tag_list": [SensorTag("Tag 1", None), SensorTag("Tag 2", None)],
        ...         "target_tag_list": [SensorTag("Tag 3", None), SensorTag("Tag 4", None)]
        ...     }
        ... )
        >>> model, metadata = builder.build()
        """
        self.name = name
        self.model_config = model_config.copy()
        self.data_config = data_config.copy()
        self.evaluation_config = evaluation_config.copy()
        self.metadata = metadata.copy()

    def build(self) -> Tuple[Optional[sklearn.base.BaseEstimator], dict]:
        """
        Build the model using the current state of the Builder

        Returns
        -------
            Tuple[Optional[sklearn.base.BaseEstimator], dict]
        """
        if self.evaluation_config.get("seed"):
            self.set_seed(seed=self.evaluation_config["seed"])

        # Get the dataset from config
        logger.debug(f"Initializing Dataset with config {self.data_config}")

        dataset = (
            self.data_config
            if isinstance(self.data_config, GordoBaseDataset)
            else _get_dataset(self.data_config)
        )

        logger.debug("Fetching training data")
        start = time.time()

        X, y = dataset.get_data()

        time_elapsed_data = time.time() - start

        # Get the model and dataset
        logger.debug(f"Initializing Model with config: {self.model_config}")
        model = serializer.pipeline_from_definition(self.model_config)

        cv_duration_sec = None

        scores: Dict[str, Any] = dict()
        if self.evaluation_config["cv_mode"].lower() in (
            "cross_val_only",
            "full_build",
        ):

            # Build up a metrics list.
            metrics_list = self.metrics_from_list(self.evaluation_config.get("metrics"))

            # Cross validate
            logger.debug("Starting cross validation")
            start = time.time()
            if hasattr(model, "predict"):

                scaler = self.evaluation_config.get("scoring_scaler")
                metrics_dict = self.build_metrics_dict(metrics_list, y, scaler=scaler)

                cv_kwargs = dict(
                    X=X,
                    y=y,
                    scoring=metrics_dict,
                    return_estimator=True,
                    cv=TimeSeriesSplit(n_splits=3),
                )
                if hasattr(model, "cross_validate"):
                    cv = model.cross_validate(**cv_kwargs)
                else:
                    cv = cross_validate(model, **cv_kwargs)

                for metric, test_metric in map(
                    lambda k: (k, f"test_{k}"), metrics_dict
                ):
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

            cv_duration_sec = time.time() - start

            # If cross_val_only, return the cv_scores and empty model.
            if self.evaluation_config["cv_mode"] == "cross_val_only":
                model = None
                self.metadata["model"] = {
                    "cross-validation": {
                        "cv-duration-sec": cv_duration_sec,
                        "scores": scores,
                    }
                }
                return model, self.metadata
        # Train
        logger.debug("Starting to train model.")
        start = time.time()
        model.fit(X, y)
        time_elapsed_model = time.time() - start

        metadata: Dict[Any, Any]
        metadata = {"user-defined": self.metadata}
        metadata["name"] = self.name
        metadata["dataset"] = dataset.get_metadata()
        utc_dt = datetime.datetime.now(datetime.timezone.utc)
        metadata["model"] = {
            "model-offset": self._determine_offset(model, X),
            "model-creation-date": str(utc_dt.astimezone()),
            "model-builder-version": __version__,
            "model-config": self.model_config,
            "data-query-duration-sec": time_elapsed_data,
            "model-training-duration-sec": time_elapsed_model,
            "cross-validation": {"cv-duration-sec": cv_duration_sec, "scores": scores},
        }

        metadata["model"].update(self._extract_metadata_from_model(model))

        return model, metadata

    def set_seed(self, seed: int):
        logger.info(f"Setting random seed: '{seed}'")
        tf.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    @staticmethod
    def build_metrics_dict(
        metrics_list: list,
        y: pd.DataFrame,
        scaler: Optional[Union[TransformerMixin, str]] = None,
    ) -> dict:

        """
        Given a list of metrics that accept a true_y and pred_y as inputs this returns a
        dictionary with keys in the form '{score}-{tag_name}' for each given target tag
        and '{score}' for the average score across all target tags and folds,
        and values being the callable make_scorer(metric_wrapper(score)). Note: score in
        {score}-{tag_name} is a sklearn's score function name with '_' replaced by '-'
        and tag_name corresponds to given target tag name with ' ' replaced by '-'.

        Parameters
        ----------
        metrics_list: list
            List of sklearn score functions
        y: pd.DataFrame
            Target data
        scaler : Optional[Union[TransformerMixin, str]]
            Scaler which will be fitted on y, and used to transform the data before
            scoring. Useful when the metrics are sensitive to the amplitude of the data, and
            you have multiple targets.


        Returns
        -------
            dict
        """
        if scaler:
            if isinstance(scaler, str) or isinstance(scaler, dict):
                scaler = serializer.pipeline_from_definition(scaler)
            logger.debug("Fitting scaler for scoring purpose")
            scaler.fit(y)

        def _score_factory(metric_func=metrics.r2_score, col_index=0):
            def _score_per_tag(y_true, y_pred):
                # This function extracts the score for each given target_tag to
                # use as scoring argument in sklearn cross_validate, as the scoring
                # must return a single value.
                if hasattr(y_true, "values"):
                    y_true = y_true.values
                if hasattr(y_pred, "values"):
                    y_pred = y_pred.values

                return metric_func(y_true[:, col_index], y_pred[:, col_index])

            return _score_per_tag

        metrics_dict = {}
        for metric in metrics_list:
            for index, col in enumerate(y.columns):
                metric_str = metric.__name__.replace("_", "-")
                metrics_dict.update(
                    {
                        metric_str
                        + f'-{col.replace(" ", "-")}': metrics.make_scorer(
                            metric_wrapper(
                                _score_factory(metric_func=metric, col_index=index),
                                scaler=scaler,
                            )
                        )
                    }
                )

            metrics_dict.update(
                {metric_str: metrics.make_scorer(metric_wrapper(metric, scaler=scaler))}
            )
        return metrics_dict

    @staticmethod
    def _determine_offset(
        model: BaseEstimator, X: Union[np.ndarray, pd.DataFrame]
    ) -> int:
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

    @staticmethod
    def _save_model_for_workflow(
        model: BaseEstimator, metadata: dict, output_dir: Union[os.PathLike, str]
    ):
        """
        Save the model according to the expected Argo workflow procedure.

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

    @staticmethod
    def _extract_metadata_from_model(
        model: BaseEstimator, metadata: dict = dict()
    ) -> dict:
        """
        Recursively check for :class:`gordo_components.model.base.GordoBase` in a
        given ``model``. If such the model exists buried inside of a
        :class:`sklearn.pipeline.Pipeline` which is then part of another
        :class:`sklearn.base.BaseEstimator`, this function will return its metadata.

        Parameters
        ----------
        model: BaseEstimator
        metadata: dict
            Any initial starting metadata, but is mainly meant to be used during
            the recursive calls to accumulate any multiple
            :class:`gordo_components.model.base.GordoBase` models found in this model

        Notes
        -----
        If there is a ``GordoBase`` model inside of a ``Pipeline`` which is not the final
        step, this function will not find it.

        Returns
        -------
        dict
            Dictionary representing accumulated calls to
            :meth:`gordo_components.model.base.GordoBase.get_metadata`
        """
        metadata = metadata.copy()

        # If it's a Pipeline, only need to get the last step, which potentially has metadata
        if isinstance(model, Pipeline):
            final_step = model.steps[-1][1]
            metadata.update(ModelBuilder._extract_metadata_from_model(final_step))
            return metadata

        # GordoBase is simple, having a .get_metadata()
        if isinstance(model, GordoBase):
            metadata.update(model.get_metadata())

        # Continue to look at object values in case, we decided to have a GordoBase
        # which also had a GordoBase as a parameter/attribute, but will satisfy BaseEstimators
        # which can take a GordoBase model as a parameter, which will then have metadata to get
        for val in model.__dict__.values():
            if isinstance(val, Pipeline):
                metadata.update(
                    ModelBuilder._extract_metadata_from_model(val.steps[-1][1])
                )
            elif isinstance(val, GordoBase) or isinstance(val, BaseEstimator):
                metadata.update(ModelBuilder._extract_metadata_from_model(val))
        return metadata

    @property
    def cache_key(self) -> str:
        """
        Calculates a hash-key from the model and data-config.

        Returns
        -------
        str:
            A 512 byte hex value as a string based on the content of the parameters.

        Examples
        -------
        >>> builder = ModelBuilder(name="My-model", model_config={"model": "something"},
        ... data_config={"tag_list": ["tag1", "tag 2"]}, evaluation_config={"cv_mode": "full_build"} )
        >>> len(builder.cache_key)
        128
        """
        if self.metadata is None:
            self.metadata = {}

        # Sets a lot of the parameters to json.dumps explicitly to ensure that we get
        # consistent hash-values even if json.dumps changes their default values
        # (and as such might generate different json which again gives different hash)
        json_rep = json.dumps(
            {
                "name": self.name,
                "model_config": self.model_config,
                "data_config": self.data_config,
                "user-defined": self.metadata,
                "evaluation_config": self.evaluation_config,
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

    def check_cache(self, model_register_dir: Union[os.PathLike, str]):
        """
        Checks if the model is cached, and returns its path if it exists.

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
        existing_model_location = disk_registry.get_value(
            model_register_dir, self.cache_key
        )

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
                f"Did not find the model with key {self.cache_key} in the register at "
                f"{model_register_dir}."
            )
            return None

    def build_with_cache(
        self,
        output_dir: Union[os.PathLike, str],
        model_register_dir: Union[os.PathLike, str] = None,
        replace_cache=False,
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
        if model_register_dir:
            logger.info(
                f"Model caching activated, attempting to read model-location with key "
                f"{self.cache_key} from register {model_register_dir}"
            )
            if replace_cache:
                logger.info(
                    "replace_cache activated, deleting any existing cache entry"
                )
                disk_registry.delete_value(model_register_dir, self.cache_key)
            else:
                cached_model_location = self.check_cache(model_register_dir)
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

        model, metadata = self.build()
        model_location = self._save_model_for_workflow(
            model=model, metadata=metadata, output_dir=output_dir
        )
        logger.info(f"Successfully built model, and deposited at {model_location}")
        if model_register_dir:
            logger.info(f"Writing model-location to model registry")
            disk_registry.write_key(model_register_dir, self.cache_key, model_location)
        return model_location

    @staticmethod
    def metrics_from_list(metric_list: Optional[List[str]] = None) -> List[Callable]:
        """
        Given a list of metric function paths. ie. sklearn.metrics.r2_score or
        simple function names which are expected to be in the ``sklearn.metrics`` module,
        this will return a list of those loaded functions.

        Parameters
        ----------
        metrics: Optional[List[str]]
            List of function paths to use as metrics for the model Defaults to
            those specified in :class:`gordo_components.workflow.config_components.NormalizedConfig`
            sklearn.metrics.explained_variance_score,
            sklearn.metrics.r2_score,
            sklearn.metrics.mean_squared_error,
            sklearn.metrics.mean_absolute_error

        Returns
        -------
        List[Callable]
            A list of the functions loaded

        Raises
        ------
        AttributeError:
           If the function cannot be loaded.
        """
        defaults = NormalizedConfig.DEFAULT_CONFIG_GLOBALS["evaluation"]["metrics"]
        funcs = list()
        for func_path in metric_list or defaults:
            func = pydoc.locate(func_path)
            if func is None:
                # Final attempt, load function from sklearn.metrics module.
                funcs.append(getattr(metrics, func_path))
            else:
                funcs.append(func)
        return funcs
