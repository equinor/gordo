# -*- coding: utf-8 -*-

import logging
import pydoc
import copy
import typing  # noqa
from typing import Union, Dict, Any, Iterable
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator


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
    Builds a branch of the tree and optionall constructs the class with the given
    leafs of the branch, if constructor_class is not none. Otherwise just the
    built leafs are returned.
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

        StepClass = pydoc.locate(
            import_str
        )  # type: Union[FeatureUnion, Pipeline, BaseEstimator]

        if StepClass is None:
            raise ImportError(f'Could not locate path: "{import_str}"')

        # FeatureUnion or another Pipeline transformer
        if any(StepClass == obj for obj in [FeatureUnion, Pipeline]):

            # Need to ensure the parameters to be supplied are valid FeatureUnion
            # & Pipeline both take a list of transformers, but with different
            # kwarg, here we pull out the list to keep _build_branch generic
            if "transformer_list" in params:
                params["transformer_list"] = _build_branch(
                    params["transformer_list"], None
                )
            elif "steps" in params:
                params["steps"] = _build_branch(params["steps"], None)

            # If params is an iterable, is has to be the first argument
            # to the StepClass (FeatureUnion / Pipeline); a list of transformers
            elif any(isinstance(params, obj) for obj in (tuple, list)):
                steps = _build_branch(params, None)
                return StepClass(steps)
            else:
                raise ValueError(
                    f"Got {StepClass} but the supplied parameters"
                    f"seem invalid: {params}"
                )

        # FunctionTransformer needs to have its `func` param loaded from
        # gordo_components
        elif StepClass == FunctionTransformer:
            for func_arg in ["func", "inverse_func"]:
                if params.get(func_arg) is not None:
                    func = pydoc.locate(params[func_arg])
                    if func is None:
                        raise ValueError(
                            f"Was unable to locate function: {params[func_arg]}"
                        )
                    params[func_arg] = func
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
