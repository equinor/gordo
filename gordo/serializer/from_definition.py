# -*- coding: utf-8 -*-

import logging
import pydoc
import copy
import typing  # noqa
from typing import Union, Dict, Any, Iterable
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator
from tensorflow.keras.models import Sequential
from .utils import validate_import_path


logger = logging.getLogger(__name__)


def import_locate(import_path: str) -> Any:
    obj = pydoc.locate(import_path)
    if obj is not None:
        if not validate_import_path(import_path):
            raise ValueError("Unsupported import path '%s'" % import_path)
    return obj


def from_definition(
    pipe_definition: Union[str, Dict[str, Dict[str, Any]]]
) -> Union[FeatureUnion, Pipeline]:
    """
    Construct a Pipeline or FeatureUnion from a definition.

    Example
    -------
    >>> import yaml
    >>> from gordo import serializer
    >>> raw_config = '''
    ... sklearn.pipeline.Pipeline:
    ...         steps:
    ...             - sklearn.decomposition.PCA:
    ...                 n_components: 3
    ...             - sklearn.pipeline.FeatureUnion:
    ...                 - sklearn.decomposition.PCA:
    ...                     n_components: 3
    ...                 - sklearn.pipeline.Pipeline:
    ...                     - sklearn.preprocessing.MinMaxScaler
    ...                     - sklearn.decomposition.TruncatedSVD:
    ...                         n_components: 2
    ...             - sklearn.ensemble.RandomForestClassifier:
    ...                 max_depth: 3
    ... '''
    >>> config = yaml.safe_load(raw_config)
    >>> scikit_learn_pipeline = serializer.from_definition(config)


    Parameters
    ---------
    pipe_definition
        List of steps for the Pipeline / FeatureUnion
    constructor_class
        What to place the list of transformers into,
        either sklearn.pipeline.Pipeline/FeatureUnion

    Returns
    -------
    sklearn.pipeline.Pipeline
        pipeline
    """
    # Avoid some mutation
    definition = copy.deepcopy(pipe_definition)
    return _build_step(definition)


def _build_branch(
    definition: Iterable[Union[str, Dict[Any, Any]]],
    constructor_class=Union[Pipeline, None],
):
    """
    Builds a branch of the tree and optionally constructs the class with the given
    leafs of the branch, if constructor_class is not none. Otherwise just the
    built leafs are returned.
    """
    steps = [_build_step(step) for step in definition]
    return steps if constructor_class is None else constructor_class(steps)


def _build_scikit_branch(
    definition: Iterable[Union[str, Dict[Any, Any]]],
    constructor_class=Union[Pipeline, None],
):
    """
    Exactly like :func:`~_build_branch` except it's expected this is going to
    be a list of tuples, where the 0th element is the name of the step.
    """
    steps = [(f"step_{i}", _build_step(step)) for i, step in enumerate(definition)]
    return steps if constructor_class is None else constructor_class(steps)


def _build_step(
    step: Union[str, Dict[str, Dict[str, Any]]]
) -> Union[FeatureUnion, Pipeline, BaseEstimator]:
    """
    Build an isolated step within a transformer list, given a dict config

    Parameters
    ----------
    step: dict/str - A dict, with a single key and associated dict
                     where the associated dict are parameters for the
                     given step.

                     Example: {'sklearn.preprocessing.PCA':
                                    {'n_components': 4}
                              }
                        Gives:  PCA(n_components=4)

                    Alternatively, 'step' can be a single string, in
                    which case the step will be initiated w/ default
                    params.

                    Example: 'sklearn.preprocessing.PCA'
                        Gives: PCA()
    Returns
    -------
        Scikit-Learn Transformer or BaseEstimator
    """
    logger.debug(f"Building step: {step}")

    # Here, 'step' _should_ be a dict with a single key
    # and an associated dict containing parameters for the desired
    # sklearn step. ie. {'sklearn.preprocessing.PCA': {'n_components': 2}}
    if isinstance(step, dict):

        if len(step.keys()) != 1:
            return _load_param_classes(step)

        import_str = list(step.keys())[0]

        StepClass: Union[FeatureUnion, Pipeline, BaseEstimator] = import_locate(
            import_str
        )

        if StepClass is None:
            raise ImportError(f'Could not locate path: "{import_str}"')

        params = step.get(import_str, dict())

        if hasattr(StepClass, "from_definition"):
            return getattr(StepClass, "from_definition")(params)

        # Load any possible classes in the params if this is a dict of maybe kwargs
        if isinstance(params, dict):
            params = _load_param_classes(params)

        # update any param values which are string locations to functions
        if isinstance(params, dict):
            for param, value in params.items():
                if isinstance(value, str):
                    possible_func = import_locate(value)
                    if callable(possible_func):
                        params[param] = possible_func

        # FeatureUnion or another Pipeline transformer
        if any(StepClass == obj for obj in [FeatureUnion, Pipeline, Sequential]):

            # Need to ensure the parameters to be supplied are valid FeatureUnion
            # & Pipeline both take a list of transformers, but with different
            # kwarg, here we pull out the list to keep _build_scikit_branch generic
            if "transformer_list" in params:
                params["transformer_list"] = _build_scikit_branch(
                    params["transformer_list"], None
                )
            elif "steps" in params:
                params["steps"] = _build_scikit_branch(params["steps"], None)

            # If params is an iterable, is has to be the first argument
            # to the StepClass (FeatureUnion / Pipeline); a list of transformers
            elif any(isinstance(params, obj) for obj in (tuple, list)):
                steps = _build_scikit_branch(params, None)
                return StepClass(steps)
            elif isinstance(params, dict) and "layers" in params:
                params["layers"] = _build_branch(params["layers"], None)
            else:
                raise ValueError(
                    f"Got {StepClass} but the supplied parameters"
                    f"seem invalid: {params}"
                )
        return StepClass(**params)

    # If step is just a string, can initialize it without any params
    # ie. "sklearn.preprocessing.PCA"
    elif isinstance(step, str):
        Step = import_locate(step)  # type: Union[FeatureUnion, Pipeline, BaseEstimator]
        if hasattr(Step, "from_definition"):
            return getattr(Step, "from_definition")({})
        else:
            return Step() if Step is not None else step

    else:
        raise ValueError(
            f"Expected step to be either a string or a dict," f"found: {type(step)}"
        )


def _build_callbacks(definitions: list):
    """
    Parameters
    ----------
    definitions: List
        List of callbacks definitions

    Examples
    --------
    >>> callbacks=_build_callbacks([{'tensorflow.keras.callbacks.EarlyStopping': {'monitor': 'val_loss,', 'patience': 10}}])
    >>> type(callbacks[0])
    <class 'tensorflow.python.keras.callbacks.EarlyStopping'>

    Returns
    -------
    dict
    """
    callbacks = []
    for callback in definitions:
        callbacks.append(_build_step(callback))
    return callbacks


def _load_param_classes(params: dict):
    """
    Inspect the params' values and determine if any can be loaded as a class.
    if so, update that param's key value as the instantiation of the class.

    Additionally, if the value of the top level is a dict, and that dict's len(.keys()) == 1
    AND that key can be loaded, it's assumed to be a class whose associated values
    should be passed in as kwargs.

    Parameters
    ----------
    params: dict
        key value pairs of kwargs, which can have full class paths defined.

    Examples
    --------
    >>> params = {"key1": "value1"}
    >>> assert _load_param_classes(params) == params  # No modifications

    # Load an actual model, without any kwargs
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> params = {"base_estimator": "sklearn.ensemble.RandomForestRegressor"}
    >>> print(_load_param_classes(params))
    {'base_estimator': RandomForestRegressor()}

    # Load an actual model, with kwargs
    >>> params = {"base_estimator": {"sklearn.ensemble.RandomForestRegressor": {"n_estimators": 20}}}
    >>> print(_load_param_classes(params))
    {'base_estimator': RandomForestRegressor(n_estimators=20)}


    Returns
    -------
    dict
        Updated params which has any possible class paths loaded up as instantiated
        objects
    """
    params = copy.copy(params)
    for key, value in params.items():

        # If value is a simple string, try to load the model/class
        if isinstance(value, str):
            Model: Union[None, BaseEstimator, Pipeline] = import_locate(value)
            if Model is not None:
                if hasattr(Model, "from_definition"):
                    params[key] = getattr(Model, "from_definition")({})
                elif isinstance(Model, type) and issubclass(Model, BaseEstimator):

                    params[key] = Model()

        # For the next bit to work, the dict must have a single key (maybe) the class path,
        # and its value must be a dict of kwargs
        elif (
            isinstance(value, dict)
            and len(value.keys()) == 1
            and isinstance(value[list(value.keys())[0]], dict)
        ):
            import_path = list(value.keys())[0]
            Model = import_locate(import_path)

            sub_params = value[import_path]

            if hasattr(Model, "from_definition"):
                params[key] = getattr(Model, "from_definition")(sub_params)
            elif Model is not None and isinstance(Model, type):

                if issubclass(Model, Pipeline) or issubclass(Model, Sequential):
                    # Model is a Pipeline, so 'value' is the definition of that Pipeline
                    # Can can just re-use the entry to building a pipeline.
                    params[key] = from_definition(value)
                else:
                    # Call this func again, incase there is nested occurances of this problem in these kwargs
                    kwargs = _load_param_classes(sub_params)
                    params[key] = Model(**kwargs)  # type: ignore
        elif key == "callbacks" and isinstance(value, list):
            params[key] = _build_callbacks(value)
    return params


def load_params_from_definition(definition: dict) -> dict:
    """
    Deserialize each value from a dictionary. Could be used for preparing kwargs for methods

    Parameters
    ----------
    definition: dict
    """
    if not isinstance(definition, dict):
        raise ValueError(
            "Expected definition to be a dict," f"found: {type(definition)}"
        )
    return _load_param_classes(definition)
