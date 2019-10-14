# -*- coding: utf-8 -*-

import logging
import pydoc
import copy
import typing  # noqa
from typing import Union, Dict, Any, Iterable
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator
from tensorflow.keras.models import Sequential


logger = logging.getLogger(__name__)


def pipeline_from_definition(
    pipe_definition: Union[str, Dict[str, Dict[str, Any]]]
) -> Union[FeatureUnion, Pipeline]:
    """
    Construct a Pipeline or FeatureUnion from a definition.

    Example
    -------
    >>> import yaml
    >>> from gordo_components import serializer
    >>> raw_config = '''
    ... sklearn.pipeline.Pipeline:
    ...         steps:
    ...             - sklearn.decomposition.pca.PCA:
    ...                 n_components: 3
    ...             - sklearn.pipeline.FeatureUnion:
    ...                 - sklearn.decomposition.pca.PCA:
    ...                     n_components: 3
    ...                 - sklearn.pipeline.Pipeline:
    ...                     - sklearn.preprocessing.data.MinMaxScaler
    ...                     - sklearn.decomposition.truncated_svd.TruncatedSVD:
    ...                         n_components: 2
    ...             - sklearn.ensemble.forest.RandomForestClassifier:
    ...                 max_depth: 3
    ... '''
    >>> config = yaml.load(raw_config)
    >>> scikit_learn_pipeline = serializer.pipeline_from_definition(config)


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
            raise ValueError(
                f"Step should have a single key, " f"found multiple: {step.keys()}"
            )

        import_str = list(step.keys())[0]
        params = step.get(import_str, dict())

        # Load any possible classes in the params if this is a dict of maybe kwargs
        if isinstance(params, dict):
            params = _load_param_classes(params)

        # update any param values which are string locations to functions
        if isinstance(params, dict):
            for param, value in params.items():
                if isinstance(value, str):
                    possible_func = pydoc.locate(value)
                    if callable(possible_func):
                        params[param] = possible_func

        StepClass: Union[FeatureUnion, Pipeline, BaseEstimator] = pydoc.locate(
            import_str
        )

        if StepClass is None:
            raise ImportError(f'Could not locate path: "{import_str}"')

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
        Step = pydoc.locate(step)  # type: Union[FeatureUnion, Pipeline, BaseEstimator]
        return Step()

    else:
        raise ValueError(
            f"Expected step to be either a string or a dict," f"found: {type(step)}"
        )


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
    >>> params = {"base_estimator": "sklearn.ensemble.forest.RandomForestRegressor"}
    >>> print(_load_param_classes(params))
    {'base_estimator': RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
                          max_features='auto', max_leaf_nodes=None,
                          min_impurity_decrease=0.0, min_impurity_split=None,
                          min_samples_leaf=1, min_samples_split=2,
                          min_weight_fraction_leaf=0.0, n_estimators='warn',
                          n_jobs=None, oob_score=False, random_state=None,
                          verbose=0, warm_start=False)}

    # Load an actual model, with kwargs
    >>> params = {"base_estimator": {"sklearn.ensemble.forest.RandomForestRegressor": {"n_estimators": 20}}}
    >>> print(_load_param_classes(params))
    {'base_estimator': RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
                          max_features='auto', max_leaf_nodes=None,
                          min_impurity_decrease=0.0, min_impurity_split=None,
                          min_samples_leaf=1, min_samples_split=2,
                          min_weight_fraction_leaf=0.0, n_estimators=20,
                          n_jobs=None, oob_score=False, random_state=None,
                          verbose=0, warm_start=False)}


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
            Model: Union[None, BaseEstimator, Pipeline] = pydoc.locate(value)
            if (
                Model is not None
                and isinstance(Model, type)
                and issubclass(Model, BaseEstimator)
            ):

                params[key] = Model()

        # For the next bit to work, the dict must have a single key (maybe) the class path,
        # and its value must be a dict of kwargs
        elif (
            isinstance(value, dict)
            and len(value.keys()) == 1
            and isinstance(value[list(value.keys())[0]], dict)
        ):
            Model = pydoc.locate(list(value.keys())[0])
            if Model is not None and isinstance(Model, type):

                if issubclass(Model, Pipeline) or issubclass(Model, Sequential):
                    # Model is a Pipeline, so 'value' is the definition of that Pipeline
                    # Can can just re-use the entry to building a pipeline.
                    params[key] = pipeline_from_definition(value)
                else:
                    # Call this func again, incase there is nested occurances of this problem in these kwargs
                    sub_params = value[list(value.keys())[0]]
                    kwargs = _load_param_classes(sub_params)
                    params[key] = Model(**kwargs)  # type: ignore
    return params
