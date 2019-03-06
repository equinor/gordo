# -*- coding: utf-8 -*-

import inspect
import logging
from typing import Iterable, Dict, Union, List  # pragma: no flakes

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer


logger = logging.getLogger(__name__)


def pipeline_into_definition(
    pipeline: Pipeline, prune_default_params: bool = False
) -> dict:
    """
    Convert an instance of ``sklearn.pipeline.Pipeline`` into a dict definition
    capable of being reconstructed with
    ``gordo_components.serializer.pipeline_from_definition``

    Parameters
    ----------
    pipeline: sklearn.pipeline.Pipeline
        Instance of pipeline to decompose
    prune_default_params: bool
        Whether to prune the default parameters found in current instance of the transformers
        vs what their default params are.

    Returns
    -------
    dict
        definitions for the pipeline, compatible to be reconstructed with
        gordo_components.pipeline_translator.pipeline_from_definition

    Example
    -------
    >>> import yaml
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.decomposition import PCA
    >>> from gordo_components.model.models import KerasAutoEncoder
    >>>
    >>> pipe = Pipeline([('pca', PCA(4)), ('ae', KerasAutoEncoder(kind='feedforward_model'))])
    >>> pipe_definition = pipeline_into_definition(pipe)  # It is now a standard python dict of primitives.
    >>> print(yaml.dump(pipe_definition))
    sklearn.pipeline.Pipeline:
      memory: null
      steps:
      - sklearn.decomposition.pca.PCA:
          copy: true
          iterated_power: auto
          n_components: 4
          random_state: null
          svd_solver: auto
          tol: 0.0
          whiten: false
      - gordo_components.model.models.KerasAutoEncoder:
          kind: feedforward_model
    <BLANKLINE>
    """
    steps = _decompose_node(pipeline, prune_default_params)
    return steps


def _decompose_node(step: object, prune_default_params: bool = False):
    """
    Decompose a specific instance of a scikit-learn transformer,
    including Pipelines or FeatureUnions

    Parameters
    ----------
    step
        An instance of a Scikit-Learn transformer class
    prune_default_params
        Whether to output the default parameter values into the definition. If True,
        only those parameters differing from the default params will be output.

    Returns
    -------
    dict
        decomposed node - Where key is the import string for the class and associated value
        is a dict of parameters for that class.
    """

    import_str = f"{step.__module__}.{step.__class__.__name__}"
    init_params = inspect.getfullargspec(step.__class__.__init__).args

    params = (
        dict()
    )  # type: Dict[str, Union[str, int, float, List[Dict[str, Dict[str, Union[str, int, float]]]]]]

    for param in [p for p in init_params if p != "self"]:

        # Can be a parameter (n_components=2) or another branch of the
        # pipeline/feature union, (steps=[{'sklearn.decomposition.PCA': {...}}, ..])
        param_val = getattr(step, param)

        # If the current step is an instance of FeatureUnion or Pipeline,
        # We'll need to decompose the sub transformers for it if it's for the
        # 'steps' input for Pipeline or 'transformer_list' for FeatureUnion
        if (
            isinstance(param_val, Iterable)
            and param in ["steps", "transformer_list"]
            and any(isinstance(step, Obj) for Obj in [FeatureUnion, Pipeline])
        ):
            params[param] = [_decompose_node(leaf[1]) for leaf in param_val]

        # Handle FunctionTransformer function object type parameters
        elif (
            isinstance(step, FunctionTransformer)
            and param in ["func", "inverse_func"]
            and callable(param_val)
        ):
            # param_val is a function for FunctionTransformer.func init param
            params[param] = f"{param_val.__module__}.{param_val.__name__}"

        else:
            params[param] = param_val
    params = _prune_default_parameters(step, params) if prune_default_params else params
    return {import_str: params}


def _prune_default_parameters(obj: object, current_params) -> dict:
    """
    Take an instance of an object and determine what the default parameters are
    against what its current parameters are.

    Parameters
    ----------
        obj: object - An instance of an object
        current_params: dict - A mapping of current parameters for the obj

    Returns
    -------
        dict - Containing only parameters which are different from default
    """

    signature = inspect.signature(obj.__class__.__init__)
    default_params = {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }
    logger.debug(f"Current params: {current_params}, default_params: {default_params}")

    return {
        k: v
        for (k, v) in current_params.items()
        if current_params[k] != default_params[k]
    }
