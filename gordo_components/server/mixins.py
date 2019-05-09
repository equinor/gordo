# -*- coding: utf-8 -*-

import logging

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


class ModelMixin:
    """
    Collection of methods for dealing with models inside of prediction views
    """

    def get_model_output(self, model: Pipeline, X: np.ndarray) -> np.ndarray:
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

    def get_transformed_input(self, model: Pipeline, X: np.ndarray):
        """
        Get the transformed input which went to the last step of the model

        Parameters
        ----------
        X: np.ndarray

        Returns
        -------
        np.ndarray
            The input to the last step of the pipeline. Often this represents
            what the input was to the model from any preprocessing steps in the
            pipeline
        """
        try:
            return model.transform(X)
        except AttributeError:
            # No transform, so X must have been the model input
            return X.values if isinstance(X, pd.DataFrame) else X

    def get_inverse_transformed_input(self, model: Pipeline, X: np.ndarray):
        """
        Get the inverse transformed input previous steps gave to the last
        step in the pipeline. Meaning that we expect X in this case to be
        transformed.

        Basically the goal is to get the original input into the pipeline, which
        is really only useful to inverse transform the output of an auto-encoder
        to get it back to comparable to the original input values before they
        were transformed before getting to the auto encoder itself.

        Parameters
        ----------
        X: np.ndarray
            Values which are assumed to have been transformed

        Returns
        -------
        np.ndarray:
            Values inverse transformed by every step in the pipeline except
            the last step/model
        """
        try:
            return model.inverse_transform(X)
        except AttributeError:
            return X.values if isinstance(X, pd.DataFrame) else X
