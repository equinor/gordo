# -*- coding: utf-8 -*-

import logging

import numpy as np

from sklearn.pipeline import Pipeline

"""
Collection of functions to work with the input and output of a scikit.base.BaseEstimator
"""

logger = logging.getLogger(__name__)


def get_model_output(model: Pipeline, X: np.ndarray) -> np.ndarray:
    """
    Get the raw output from the current model given X.
    Will try to `predict` and then `transform`, raising an error
    if both fail.

    Parameters
    ----------
    X: np.ndarray
        2d array of sample(s)

    Returns
    -------
    np.ndarray
        The raw output of the model in numpy array form.
    """
    try:
        return model.predict(X)  # type: ignore

    # Model may only be a transformer
    except AttributeError:
        try:
            return model.transform(X)  # type: ignore
        except Exception as exc:
            logger.error(f"Failed to predict or transform; error: {exc}")
            raise
