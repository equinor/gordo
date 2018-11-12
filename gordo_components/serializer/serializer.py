# -*- coding: utf-8 -*-

import bz2
import glob
import json
import logging
import os
import pydoc
import re
import pickle

from os import path
from typing import Tuple, Union

from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)

N_STEP_REGEX = re.compile(r'.*n_step=([0-9]+)')
CLASS_REGEX  = re.compile(r'.*class=(.*$)')


def load(source_dir: str) -> object:
    """
    Load an object from a directory, saved by
    gordo_components.serializer.pipeline_serializer.dump


    This take a directory, which is either top-level, meaning it contains
    a sub directory in the naming scheme: "n_step=<int>-class=<path.to.Class>"
    or the aforementioned naming scheme directory directly. Will return that
    unsterilized object.


    Parameters
    ----------
        source_dir: str - Location of the top level dir the pipeline was saved

    Returns
    -------
        object
    """
    # This source dir should have a single pipeline entry directory.
    # may have been passed a top level dir, containing such an entry:
    if not source_dir.startswith('n_step'):
        dirs = [d for d in os.listdir(source_dir)
                if 'n_step=' in d]
        if len(dirs) != 1:
            raise ValueError(f'Found multiple object entries to load from, '
                             f'should pass a directory to pipeline directly or '
                             f'a directory containing a single object entry.'
                             f'Possible objects found: {dirs}')
        else:
            source_dir = path.join(source_dir, dirs[0])

    # Load step always returns a tuple of (str, object), index to object
    return _load_step(source_dir)[1]


def _parse_dir_name(source_dir: str) -> Tuple[int, str]:
    """
    Parses the required params from a directory name for loading
    Expected name format "n_step=<int>-class=<path.to.class.Model>"
    """
    n_step = N_STEP_REGEX.search(source_dir)
    if n_step is None:
        raise ValueError(f'Source dir not valid, expected "n_step=" in '
                         f'directory but instead got: {source_dir}')
    else:
        n_step = int(n_step.groups()[0])

    class_path = CLASS_REGEX.search(source_dir)
    if class_path is None:
        raise ValueError(f'Source dir not valid, expected "class=" in directory '
                         f'but instead got: {source_dir}')
    else:
        class_path = class_path.groups()[0]
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
    StepClass = pydoc.locate(class_path)
    if StepClass is None:
        logger.warning(f'Specified a class path of "{class_path}" but it does '
                       f'not exist. Will attempt to unpickle it from file in '
                       f'source directory: {source_dir}.')
    step_name = f'step={str(n_step).zfill(3)}'
    params = dict()

    # If this is a FeatureUnion, we also have a `params.json` for it
    if StepClass == FeatureUnion:
        with open(os.path.join(source_dir, 'params.json'), 'r') as f:
            params = json.load(f)

    # Pipelines and FeatureUnions have sub steps which need to be loaded
    if any(StepClass == Obj for Obj in (Pipeline, FeatureUnion)):

        # Load the sub_dirs to load into the Pipeline/FeatureUnion in order
        sub_dirs_to_load = sorted([sub_dir for sub_dir in os.listdir(source_dir)
                                   if path.isdir(path.join(source_dir, sub_dir))],
                                  key=lambda d: _parse_dir_name(d)[0])
        steps = [_load_step(path.join(source_dir, sub_dir))
                 for sub_dir in sub_dirs_to_load]
        return step_name, StepClass(steps, **params)

    # May model implementing load_from_dir method, from GordoBaseModel
    elif hasattr(StepClass, 'load_from_dir'):
        return step_name, StepClass.load_from_dir(source_dir)

    # Otherwise we have a normal Scikit-Learn transformer
    else:
        # Find the name of this file in the directory, should only be one
        file = glob.glob(path.join(source_dir, '*.pkl.gz'))
        if len(file) != 1:
            raise ValueError(f'Expected a single file in what is expected to be '
                             f'a single object directory, found {len(file)} '
                             f'in directory: {source_dir}')
        with bz2.open(path.join(source_dir, file[0]), 'rb') as f:
            return step_name, pickle.load(f)


def dump(obj: object, dest_dir: str):
    """
    Serialize an object into a directory

    The object must either be picklable or implement BOTH a `save_to_dir` AND
    `load_from_dir` methods. This object can hold multiple objects, specifically
    it can be a sklearn.pipeline.[FeatureUnion, Pipeline] object, in such a case
    it's sub transformers (steps/transformer_list) will be serialized recursively.

    Example:

    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.preprocessing import PCA
    >>> from gordo_components.model.models import KerasModel
    >>> from gordo_components import serializer
    >>> pipe = Pipeline([
    ...     # PCA is picklable
    ...     ('pca', PCA(3)),
    ...     # KerasModel implements both `save_to_dir` and `load_from_dir`
    ...     ('model', KerasModel(kind='feedforward_symetric'))
    ... ])
    >>> serializer.dump(obj=pipe, dest_dir='/my-model')
    >>> pipe_clone = serializer.load(source_dir='/my-model')

    Parameters
    ----------
        obj: object - The object which to dump. Must be picklable or implement
                      a `save_to_dir` AND `load_from_dir` method.
    Returns
    -------
        None
    """
    _dump_step(step=('obj', obj), n_step=0, dest_dir=dest_dir)


def _dump_step(step: Tuple[str, object], dest_dir: str, n_step: int=0):
    """
    Accepts any Scikit-Learn transformer and dumps it into a directory
    recoverable by gordo_components.serializer.pipeline_serializer.load

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
        for i, step in enumerate(getattr(step_transformer, steps_attr)):
            _dump_step(step=step, n_step=i, dest_dir=sub_dir)

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
            if not hasattr(step_transformer, 'load_from_dir'):
                raise AttributeError(
                    f'The object in this step implements a "save_to_dir" but '
                    f'not "load_from_dir" it will be un-recoverable!')
            logger.info(f'Saving model to sub_dir: {sub_dir}')
            step_transformer.save_to_dir(os.path.join(sub_dir))
        else:
            with bz2.open(os.path.join(sub_dir, f'{step_name}.pkl.gz'), 'wb') as f:
                pickle.dump(step_transformer, f)
