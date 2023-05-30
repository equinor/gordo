# -*- coding: utf-8 -*-

import inspect
import logging

from sklearn.pipeline import Pipeline


logger = logging.getLogger(__name__)


def into_definition(
    pipeline: Pipeline, prune_default_params: bool = False, tuples_to_list: bool = True
) -> dict:
    """
    Convert an instance of ``sklearn.pipeline.Pipeline`` into a dict definition
    capable of being reconstructed with
    ``gordo.serializer.from_definition``

    Parameters
    ----------
    pipeline
        Instance of pipeline to decompose
    prune_default_params
        Whether to prune the default parameters found in current instance of the transformers
        vs what their default params are.
    tuples_to_list
        Convert all tuples in output to lists

    Returns
    -------
        definitions for the pipeline, compatible to be reconstructed with
        :func:`gordo.serializer.from_definition`

    Example
    -------
    >>> import yaml
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.decomposition import PCA
    >>> from gordo.machine.model.models import KerasAutoEncoder
    >>>
    >>> pipe = Pipeline([('pca', PCA(4)), ('ae', KerasAutoEncoder(kind='feedforward_model'))])
    >>> pipe_definition = into_definition(pipe)  # It is now a standard python dict of primitives.
    >>> print(yaml.dump(pipe_definition))
    sklearn.pipeline.Pipeline:
      memory: null
      steps:
      - sklearn.decomposition._pca.PCA:
          copy: true
          iterated_power: auto
          n_components: 4
          n_oversamples: 10
          power_iteration_normalizer: auto
          random_state: null
          svd_solver: auto
          tol: 0.0
          whiten: false
      - gordo.machine.model.models.KerasAutoEncoder:
          kind: feedforward_model
      verbose: false
    <BLANKLINE>
    """
    steps = _decompose_node(pipeline, prune_default_params, tuples_to_list)
    return steps


def _decompose_node(
    step: object, prune_default_params: bool = False, tuples_to_list: bool = True
):
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
    tuples_to_list
        Convert all tuples in output to lists

    Returns
    -------
        decomposed node - Where key is the import string for the class and associated value
        is a dict of parameters for that class.
    """

    import_str = f"{step.__module__}.{step.__class__.__name__}"

    if hasattr(step, "into_definition"):
        definition = getattr(step, "into_definition")()
    else:
        params = getattr(step, "get_params")(deep=False)

        definition = load_definition_from_params(params)

        definition = (
            _prune_default_parameters(step, definition)
            if prune_default_params
            else definition
        )

    if prune_default_params:
        new_definition = {}
        for k, v in definition:
            if isinstance(v, tuple):
                v = list(v)
            new_definition[k] = v
        definition = new_definition
    return {import_str: definition}


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


def load_definition_from_params(params: dict, tuples_to_list: bool = True) -> dict:
    """
    Recursively decomposing each of values from params into the definition

    Parameters
    ----------
        params: dict
        tuples_to_list: bool

    Returns
    -------
        dict

    """
    definition = {}
    for param, param_val in params.items():

        if hasattr(param_val, "get_params") or hasattr(param_val, "into_definition"):
            definition[param] = _decompose_node(
                param_val, tuples_to_list=tuples_to_list
            )

        # Handle parameter value that is a list
        elif isinstance(param_val, list):

            # Decompose second elements; these are tuples of (str, BaseEstimator)
            # or list of other types such as ints.
            # TODO: Make this more robust, probably via another function to parse the iterable recursively
            # TODO: b/c it _could_, in theory, be a dict of {str: BaseEstimator} or similar.
            definition[param] = [
                _decompose_node(leaf[1], tuples_to_list=tuples_to_list)
                if isinstance(leaf, tuple)
                else leaf
                for leaf in param_val
            ]

        # Handle FunctionTransformer function object type parameters
        elif callable(param_val):
            # param_val is a function for FunctionTransformer.func init param
            definition[param] = f"{param_val.__module__}.{param_val.__name__}"

        else:
            definition[param] = param_val
    return definition
