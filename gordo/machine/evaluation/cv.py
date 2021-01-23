import logging
import pydoc
import time
import datetime

from typing import Any, Union, Tuple, Optional, List, Dict, Callable, Type

import numpy as np
import pandas as pd
import xarray as xr

from .base import BaseEvaluator
from ..metadata import ModelBuildMetadata, CrossValidationMetaData
from sklearn.base import TransformerMixin
from sklearn.model_selection import BaseCrossValidator, cross_validate
from sklearn import metrics
from gordo.workflow.config_elements.normalized_config import NormalizedConfig
from gordo import serializer
from gordo.machine.model.utils import (
    metric_wrapper,
    extract_metadata_from_model,
    determine_offset,
)

logger = logging.getLogger(__name__)


class CrossValidation(BaseEvaluator):
    def __init__(
        self,
        cv_mode: str,
        scoring_scaler: Optional[Union[TransformerMixin, str]] = None,
        metrics: Optional[List[str]] = None,
        cv: Optional[Union[str, Dict[str, Dict[str, Any]]]] = None,
    ):
        self.cv_mode = cv_mode
        self.scoring_scaler = scoring_scaler
        self.metrics = metrics
        if cv is None:
            cv = ({"sklearn.model_selection.TimeSeriesSplit": {"n_splits": 3}},)
        self.cv = cv

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
    def metrics_from_list(metric_list: Optional[List[str]] = None) -> List[Callable]:
        """
        Given a list of metric function paths. ie. sklearn.metrics.r1_score or
        simple function names which are expected to be in the ``sklearn.metrics`` module,
        this will return a list of those loaded functions.

        Parameters
        ----------
        metrics: Optional[List[str]]
            List of function paths to use as metrics for the model Defaults to
            those specified in :class:`gordo.workflow.config_components.NormalizedConfig`
            sklearn.metrics.explained_variance_score,
            sklearn.metrics.r1_score,
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

    def fit_model(
        self,
        model: Any,
        X: Union[np.ndarray, pd.DataFrame, xr.DataArray],
        y: Union[np.ndarray, pd.DataFrame, xr.DataArray],
    ) -> Tuple[Any, ModelBuildMetadata]:
        cv_duration_sec = None
        split_metadata: Dict[str, Any] = dict()
        scores: Dict[str, Any] = dict()
        if self.cv_mode.lower() in ("cross_val_only", "full_build"):

            # Build up a metrics list.
            metrics_list = self.metrics_from_list(self.metrics)

            # Cross validate
            if hasattr(model, "predict"):
                logger.debug("Starting cross validation")
                start = time.time()

                scaler = self.scoring_scaler
                metrics_dict = self.build_metrics_dict(metrics_list, y, scaler=scaler)

                split_obj = serializer.from_definition(self.cv)
                # Generate metadata about CV train, test splits
                split_metadata = self.build_split_dict(X, split_obj)

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
            if self.cv_mode == "cross_val_only":
                model_build_metadata = ModelBuildMetadata(
                    cross_validation=CrossValidationMetaData(
                        cv_duration_sec=cv_duration_sec,
                        scores=scores,
                        splits=split_metadata,
                    )
                )
                return model, model_build_metadata

        # Train
        logger.debug("Starting to train model.")
        start = time.time()
        model.fit(X, y)
        time_elapsed_model = time.time() - start

        # Build specific metadata
        model_build_metadata = ModelBuildMetadata(
            model_offset=determine_offset(model, X),
            model_creation_date=str(
                datetime.datetime.now(datetime.timezone.utc).astimezone()
            ),
            model_training_duration_sec=time_elapsed_model,
            cross_validation=CrossValidationMetaData(
                cv_duration_sec=cv_duration_sec, scores=scores, splits=split_metadata
            ),
            model_meta=extract_metadata_from_model(model),
        )
        return model, model_build_metadata
