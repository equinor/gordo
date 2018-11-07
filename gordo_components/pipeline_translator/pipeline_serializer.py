# -*- coding: utf-8 -*-

import os
import joblib
import bz2
import json
import logging

from typing import Tuple

from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.base import BaseEstimator


logger = logging.getLogger(__name__)


def dump(pipe: Pipeline, dest_dir: str):
    """
    Serialize a Pipeline into a directory

    Parameters
    ----------
        pipe: sklearn.pipeline.Pipeline - The initialized pipeline to store
                                          into a directory
    Returns
    -------
        None
    """
    _dump_step(step=('pipe', pipe), n_step=0, dest_dir=dest_dir)


def _dump_step(step: Tuple[str, BaseEstimator], dest_dir: str, n_step: int=0):
    """
    Accepts any Scikit-Learn transformer and dumps it into a directory
    recoverable by gordo_components.pipeline_translator.pipeline_serializer.load

    Parameters
    ----------
        step: Tuple[str, sklearn.base.BaseEstimator] - The step to dump
        dest_dir: str - The path to the top level directory to start the
                        potentially recursive saving of steps.
        n_step: int - The order of this step in the pipeline, default to 0

    Returns
    -------
        None - Creates a new directory at the `dest_dir` in the format:
               `n_step=000-class=<full.path.to.Object` with any required files
               for recovery stored there.
    """
    step_name, step_transformer = step
    step_import_str = f'{step_transformer.__module__}.{step_transformer.__class__.__name__}'
    sub_dir = os.path.join(dest_dir, f'n_step={str(n_step).zfill(3)}-class={step_import_str}')

    os.makedirs(sub_dir, exist_ok=True)

    if any(isinstance(step_transformer, Obj) for Obj in [FeatureUnion, Pipeline]):
        steps_attr = 'transformer_list' if isinstance(step_transformer, FeatureUnion) else 'steps'
        [_dump_step(step=step, n_step=i, dest_dir=sub_dir)
         for i, step in enumerate(getattr(step_transformer, steps_attr))]

        # If this is a feature Union, we want to save `n_jobs` & `transformer_weights`
        if isinstance(step_transformer, FeatureUnion):
            params = {
                'n_jobs'             : getattr(step_transformer, 'n_jobs'),
                'transformer_weights': getattr(step_transformer, 'transformer_weights')
            }
            with open(os.path.join(sub_dir, 'params.json'), 'w') as f:
                json.dump(params, f)
    else:
        if hasattr(step_transformer, 'save_to_dir'):
            logger.info(f'Saving model to sub_dir: {sub_dir}')
            step_transformer.save_to_dir(os.path.join(sub_dir))
        else:
            with bz2.open(os.path.join(sub_dir, f'{step_name}.pkl.gz'), 'wb') as f:
                joblib.dump(step, f)
