# -*- coding: utf-8 -*-

import datetime
import hashlib
import json
import pydoc
import logging
import os
import time
import random
from pathlib import Path
from typing import Union, Optional, Dict, Any, Tuple, Type, List, Callable

import pandas as pd
import numpy as np
import tensorflow as tf
import xarray as xr

import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import metrics
from sklearn.model_selection import BaseCrossValidator, cross_validate
from sklearn.pipeline import Pipeline

from gordo.util import disk_registry
from gordo import (
    serializer,
    __version__,
    MAJOR_VERSION,
    MINOR_VERSION,
    IS_UNSTABLE_VERSION,
)
from gordo_dataset.dataset import _get_dataset
from gordo.machine.model.base import GordoBase
from gordo.machine.model.utils import metric_wrapper
from gordo.workflow.config_elements.normalized_config import NormalizedConfig
from gordo.machine import Machine
from gordo.machine.metadata import (
    BuildMetadata,
    ModelBuildMetadata,
    DatasetBuildMetadata,
    CrossValidationMetaData,
)


logger = logging.getLogger(__name__)


class ModelBuilder:
    def __init__(self, machine: Machine):
        """
        Build a model for a given :class:`gordo.workflow.config_elements.machine.Machine`

        Parameters
        ----------
        machine: Machine

        Example
        -------
        >>> from gordo_dataset.sensor_tag import SensorTag
        >>> from gordo.machine import Machine
        >>> machine = Machine(
        ...     name="special-model-name",
        ...     model={"sklearn.decomposition.PCA": {"svd_solver": "auto"}},
        ...     dataset={
        ...         "type": "RandomDataset",
        ...         "train_start_date": "2017-12-25 06:00:00Z",
        ...         "train_end_date": "2017-12-30 06:00:00Z",
        ...         "tag_list": [SensorTag("Tag 1", None), SensorTag("Tag 2", None)],
        ...         "target_tag_list": [SensorTag("Tag 3", None), SensorTag("Tag 4", None)]
        ...     },
        ...     project_name='test-proj',
        ... )
        >>> builder = ModelBuilder(machine=machine)
        >>> model, machine = builder.build()
        """
        # Avoid overwriting the passed machine, copy doesn't work if it holds
        # reference to a loaded Tensorflow model; .to_dict() serializes it to
        # a primitive dict representation.
        self.machine = Machine(**machine.to_dict())

    @property
    def cached_model_path(self) -> Union[os.PathLike, str, None]:
        return getattr(self, "_cached_model_path", None)

    @cached_model_path.setter
    def cached_model_path(self, value):
        self._cached_model_path = value

    def build(
        self,
        output_dir: Optional[Union[os.PathLike, str]] = None,
        model_register_dir: Optional[Union[os.PathLike, str]] = None,
        replace_cache=False,
    ) -> Tuple[sklearn.base.BaseEstimator, Machine]:
        """
        Always return a model and its metadata.

        If ``output_dir`` is supplied, it will save the model there.
        ``model_register_dir`` points to the model cache directory which it will
        attempt to read the model from. Supplying both will then have the effect
        of both; reading from the cache and saving that cached model to the new
        output directory.

        Parameters
        ----------
        output_dir: Optional[Union[os.PathLike, str]]
            A path to where the model will be deposited.
        model_register_dir: Optional[Union[os.PathLike, str]]
            A path to a register, see `:func:gordo.util.disk_registry`.
            If this is None then always build the model, otherwise try to resolve
            the model from the registry.
        replace_cache: bool
            Forces a rebuild of the model, and replaces the entry in the cache
            with the new model.

        Returns
        -------
        Tuple[sklearn.base.BaseEstimator, Machine]
            Built model and an updated ``Machine``
        """
        if not model_register_dir:
            model, machine = self._build()
        else:
            logger.debug(
                f"Model caching activated, attempting to read model-location with key "
                f"{self.cache_key} from register {model_register_dir}"
            )
            self.cached_model_path = self.check_cache(model_register_dir)

            if replace_cache:
                logger.info("replace_cache=True, deleting any existing cache entry")
                disk_registry.delete_value(model_register_dir, self.cache_key)
                self.cached_model_path = None

            # Load the model from previous cached directory
            if self.cached_model_path:
                model = serializer.load(self.cached_model_path)
                metadata = serializer.load_metadata(self.cached_model_path)
                metadata["metadata"][
                    "user_defined"
                ] = self.machine.metadata.user_defined

                metadata["runtime"] = self.machine.runtime

                machine = Machine(**metadata)

            # Otherwise build and cache the model
            else:
                model, machine = self._build()
                self.cached_model_path = self._save_model(
                    model=model, machine=machine, output_dir=output_dir  # type: ignore
                )
                logger.info(f"Built model, and deposited at {self.cached_model_path}")
                logger.info(f"Writing model-location to model registry")
                disk_registry.write_key(  # type: ignore
                    model_register_dir, self.cache_key, self.cached_model_path
                )

        # Save model to disk, if we're not building for cv only purposes.
        if output_dir and (self.machine.evaluation.get("cv_mode") != "cross_val_only"):
            self.cached_model_path = self._save_model(
                model=model, machine=machine, output_dir=output_dir
            )
        return model, machine

    def _build(self) -> Tuple[sklearn.base.BaseEstimator, Machine]:
        """
        Build the model using the current state of the Builder

        Returns
        -------
            Tuple[sklearn.base.BaseEstimator, dict]
        """
        # Enforce random seed to 0 if not specified.
        self.set_seed(seed=self.machine.evaluation.get("seed", 0))

        # Get the dataset from config
        logger.debug(
            f"Initializing Dataset with config {self.machine.dataset.to_dict()}"
        )

        dataset = _get_dataset(self.machine.dataset.to_dict())

        logger.debug("Fetching training data")
        start = time.time()

        X, y = dataset.get_data()

        time_elapsed_data = time.time() - start

        # Get the model and dataset
        logger.debug(f"Initializing Model with config: {self.machine.model}")
        model = serializer.from_definition(self.machine.model)

        cv_duration_sec = None

        machine: Machine = Machine(
            name=self.machine.name,
            dataset=self.machine.dataset.to_dict(),
            metadata=self.machine.metadata,
            model=self.machine.model,
            project_name=self.machine.project_name,
            evaluation=self.machine.evaluation,
            runtime=self.machine.runtime,
        )

        split_metadata: Dict[str, Any] = dict()
        scores: Dict[str, Any] = dict()
        if self.machine.evaluation["cv_mode"].lower() in (
            "cross_val_only",
            "full_build",
        ):

            # Build up a metrics list.
            metrics_list = self.metrics_from_list(
                self.machine.evaluation.get("metrics")
            )

            # Cross validate
            if hasattr(model, "predict"):
                logger.debug("Starting cross validation")
                start = time.time()

                scaler = self.machine.evaluation.get("scoring_scaler")
                metrics_dict = self.build_metrics_dict(metrics_list, y, scaler=scaler)

                split_obj = serializer.from_definition(
                    self.machine.evaluation.get(
                        "cv",
                        {"sklearn.model_selection.TimeSeriesSplit": {"n_splits": 3}},
                    )
                )
                # Generate metadata about CV train, test splits
                split_metadata = ModelBuilder.build_split_dict(X, split_obj)

                cv_kwargs = dict(
                    X=X, y=y, scoring=metrics_dict, return_estimator=True, cv=split_obj
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

                cv_duration_sec = time.time() - start
            else:
                logger.debug("Unable to score model, has no attribute 'predict'.")

            # If cross_val_only, return without fitting to the whole dataset
            if self.machine.evaluation["cv_mode"] == "cross_val_only":
                machine.metadata.build_metadata = BuildMetadata(
                    model=ModelBuildMetadata(
                        cross_validation=CrossValidationMetaData(
                            cv_duration_sec=cv_duration_sec,
                            scores=scores,
                            splits=split_metadata,
                        )
                    ),
                    dataset=DatasetBuildMetadata(
                        query_duration_sec=time_elapsed_data,
                        dataset_meta=dataset.get_metadata(),
                    ),
                )
                return model, machine

        # Train
        logger.debug("Starting to train model.")
        start = time.time()
        model.fit(X, y)
        time_elapsed_model = time.time() - start

        # Build specific metadata
        machine.metadata.build_metadata = BuildMetadata(
            model=ModelBuildMetadata(
                model_offset=self._determine_offset(model, X),
                model_creation_date=str(
                    datetime.datetime.now(datetime.timezone.utc).astimezone()
                ),
                model_builder_version=__version__,
                model_training_duration_sec=time_elapsed_model,
                cross_validation=CrossValidationMetaData(
                    cv_duration_sec=cv_duration_sec,
                    scores=scores,
                    splits=split_metadata,
                ),
                model_meta=self._extract_metadata_from_model(model),
            ),
            dataset=DatasetBuildMetadata(
                query_duration_sec=time_elapsed_data,
                dataset_meta=dataset.get_metadata(),
            ),
        )
        return model, machine

    def set_seed(self, seed: int):
        logger.info(f"Setting random seed: '{seed}'")
        tf.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    @staticmethod
    def build_split_dict(X: pd.DataFrame, split_obj: Type[BaseCrossValidator]) -> dict:
        """
        Get dictionary of cross-validation training dataset split metadata

        Parameters
        ----------
        X: pd.DataFrame
            The training dataset that will be split during cross-validation.
        split_obj: Type[sklearn.model_selection.BaseCrossValidator]
            The cross-validation object that returns train, test indices for splitting.

        Returns
        -------
        split_metadata: Dict[str,Any]
            Dictionary of cross-validation train/test split metadata
        """
        split_metadata: Dict[str, Any] = dict()
        for i, (train_ind, test_ind) in enumerate(split_obj.split(X)):
            split_metadata.update(
                {
                    f"fold-{i+1}-train-start": X.index[train_ind[0]],
                    f"fold-{i+1}-train-end": X.index[train_ind[-1]],
                    f"fold-{i+1}-test-start": X.index[test_ind[0]],
                    f"fold-{i+1}-test-end": X.index[test_ind[-1]],
                }
            )
            split_metadata.update({f"fold-{i+1}-n-train": len(train_ind)})
            split_metadata.update({f"fold-{i+1}-n-test": len(test_ind)})
        return split_metadata

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
                scaler = serializer.from_definition(scaler)
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
        if isinstance(X, pd.DataFrame) or isinstance(X, xr.DataArray):
            X = X.values
        out = model.predict(X) if hasattr(model, "predict") else model.transform(X)
        return len(X) - len(out)

    @staticmethod
    def _save_model(
        model: BaseEstimator,
        machine: Union[Machine, dict],
        output_dir: Union[os.PathLike, str],
    ):
        """
        Save the model according to the expected Argo workflow procedure.

        Parameters
        ----------
        model: BaseEstimator
            The model to save to the directory with gordo serializer.
        machine: Union[Machine, dict]
            Machine instance used to build this model.
        output_dir: Union[os.PathLike, str]
            The directory where to save the model, will create directories if needed.

        Returns
        -------
        Union[os.PathLike, str]
            Path to the saved model
        """
        os.makedirs(output_dir, exist_ok=True)  # Ok if some dirs exist
        serializer.dump(
            model,
            output_dir,
            metadata=machine.to_dict() if isinstance(machine, Machine) else machine,
        )
        return output_dir

    @staticmethod
    def _extract_metadata_from_model(
        model: BaseEstimator, metadata: dict = dict()
    ) -> dict:
        """
        Recursively check for :class:`gordo.machine.model.base.GordoBase` in a
        given ``model``. If such the model exists buried inside of a
        :class:`sklearn.pipeline.Pipeline` which is then part of another
        :class:`sklearn.base.BaseEstimator`, this function will return its metadata.

        Parameters
        ----------
        model: BaseEstimator
        metadata: dict
            Any initial starting metadata, but is mainly meant to be used during
            the recursive calls to accumulate any multiple
            :class:`gordo.machine.model.base.GordoBase` models found in this model

        Notes
        -----
        If there is a ``GordoBase`` model inside of a ``Pipeline`` which is not the final
        step, this function will not find it.

        Returns
        -------
        dict
            Dictionary representing accumulated calls to
            :meth:`gordo.machine.model.base.GordoBase.get_metadata`
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
        return self.calculate_cache_key(self.machine)

    @staticmethod
    def calculate_cache_key(machine: Machine) -> str:
        """
                Calculates a hash-key from the model and data-config.

                Returns
                -------
                str:
                    A 512 byte hex value as a string based on the content of the parameters.

                Examples
                -------
                >>> from gordo.machine import Machine
                >>> from gordo_dataset.sensor_tag import SensorTag
                >>> machine = Machine(
                ...     name="special-model-name",
                ...     model={"sklearn.decomposition.PCA": {"svd_solver": "auto"}},
                ...     dataset={
                ...         "type": "RandomDataset",
                ...         "train_start_date": "2017-12-25 06:00:00Z",
                ...         "train_end_date": "2017-12-30 06:00:00Z",
                ...         "tag_list": [SensorTag("Tag 1", None), SensorTag("Tag 2", None)],
                ...         "target_tag_list": [SensorTag("Tag 3", None), SensorTag("Tag 4", None)]
                ...     },
                ...     project_name='test-proj'
                ... )
                >>> builder = ModelBuilder(machine)
                >>> len(builder.cache_key)
                128
                """
        # Sets a lot of the parameters to json.dumps explicitly to ensure that we get
        # consistent hash-values even if json.dumps changes their default values
        # (and as such might generate different json which again gives different hash)
        json_rep = json.dumps(
            {
                "name": machine.name,
                "model_config": machine.model,
                "data_config": machine.dataset.to_dict(),
                "evaluation_config": machine.evaluation,
                "gordo-major-version": MAJOR_VERSION,
                "gordo-minor-version": MINOR_VERSION,
                "is-gordo-unstable": IS_UNSTABLE_VERSION,
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
            those specified in :class:`gordo.workflow.config_components.NormalizedConfig`
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
