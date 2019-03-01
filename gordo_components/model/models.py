# -*- coding: utf-8 -*-

import logging
import json
from typing import Union, Callable, Dict, Any, Optional
from os import path
from contextlib import contextmanager

import keras.backend as K
import numpy as np
import pandas as pd

from sklearn.base import TransformerMixin
from sklearn.metrics import explained_variance_score
from sklearn.exceptions import NotFittedError
from keras.wrappers.scikit_learn import BaseWrapper
import keras.models
from keras.models import load_model
from gordo_components.model.base import GordoBase

# This is required to run `register_model_builder` against registered factories
from gordo_components.model.factories import *  # pragma: no flakes

from gordo_components.model.register import register_model_builder


logger = logging.getLogger(__name__)


@contextmanager
def possible_tf_mgmt(keras_model):
    """
    When serving a Keras model backed by tensorflow, the object is expected
    to have `_tf_graph` and `_tf_session` stored attrs. Which will be used
    as the default when calling the Keras model
    """
    logger.info(f"Keras backend: {K.backend()}")
    if K.backend() == "tensorflow":
        logger.debug(f"Using keras_model {keras_model} local TF Graph and Session")
        with keras_model._tf_graph.as_default(), keras_model._tf_session.as_default():
            yield
    else:
        yield


class KerasBaseEstimator(BaseWrapper, GordoBase):
    def __init__(
        self,
        kind: Union[str, Callable[[int, Dict[str, Any]], keras.models.Model]],
        **kwargs,
    ) -> None:
        """
        Initialized a Scikit-Learn API compatitble Keras model with a pre-registered function or a builder function
        directly.

        Example use:
        ```
        from gordo_components.model import models


        def this_function_returns_a_special_keras_model(n_features, extra_param1, extra_param2):
            ...

        scikit_based_transformer = KerasAutoEncoder(kind=this_function_returns_a_special_keras_model,
                                        extra_param1='special_parameter',
                                        extra_param2='another_parameter')

        scikit_based_transformer.fit(X, y)
        scikit_based_transformer.transform(X)
        ```

        kind: Union[callable, str] - The structure of the model to build. As designated by any registered builder
                                     functions, registered with gordo_compontents.model.register.register_model_builder
                                     Alternatively, one may pass a builder function directly to this argument. Such a
                                     function should accept `n_features` as it's first argument, and pass any additional
                                     parameters to `**kwargs`

        kwargs: dict -  Any additional args which are passed to the factory 
                        building function and/or any additional args to be passed
                        to Keras' fit() method
        """
        # Tensorflow requires managed graph/session as to not default to global
        if K.backend() == "tensorflow":
            logger.info(f"Keras backend detected as tensorflow, keeping local graph")
            import tensorflow as tf

            self._tf_graph = tf.Graph()
            self._tf_session = tf.Session(graph=self._tf_graph)
        else:
            logger.info(f"Keras backend detected as NOT tensorflow, but: {K.backend()}")
            self._tf_session = None
            self._tf_graph = None

        self.build_fn = None
        self.kwargs = kwargs

        class_name = self.__class__.__name__

        if callable(kind):
            register_model_builder(type=class_name)(kind)
            self.kind = kind.__name__
        else:
            if kind not in register_model_builder.factories[class_name]:
                raise ValueError(
                    f"kind: {kind} is not an available model for type: {class_name}!"
                )
            self.kind = kind

    @property
    def sk_params(self):
        return self.kwargs

    def fit(self, X, y, sample_weight=None, **kwargs):
        logger.debug(f"Fitting to data of length: {len(X)}")
        self.kwargs.update({"n_features": X.shape[1]})
        with possible_tf_mgmt(self):
            super().fit(X, y, sample_weight=None, **kwargs)
        return self

    def get_params(self, **params):
        params = super().get_params(**params)
        params.update({"kind": self.kind})
        return params

    def __call__(self):
        build_fn = register_model_builder.factories[self.__class__.__name__][self.kind]
        with possible_tf_mgmt(self):
            return build_fn(**self.sk_params)

    def save_to_dir(self, directory: str):
        params = self.get_params()
        with open(path.join(directory, "params.json"), "w") as f:
            json.dump(params, f)
        if hasattr(self, "model") and self.model is not None:
            with possible_tf_mgmt(self):
                self.model.save(path.join(directory, "model.h5"))

    def score(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> float:
        """
        Returns the appropriate scoring metric for a given model

        Parameters
        ----------
        X: Union[np.ndarray, pd.DataFrame] - Input data to the model
        y: Union[np.ndarray, pd.DataFrame] - Target
        sample_weight: Optional[np.ndarray] - sample weights

        Returns
        -------
        score: float
        """
        raise NotImplementedError(
            f"Subclasses of {self.__class__.__name__} must implement the .score(...) method."
        )

    @classmethod
    def load_from_dir(cls, directory: str):
        with open(path.join(directory, "params.json"), "r") as f:
            params = json.load(f)
        obj = cls(**params)
        model_file = path.join(directory, "model.h5")
        if path.isfile(model_file):
            with possible_tf_mgmt(obj):
                K.set_learning_phase(0)
                obj.model = load_model(model_file)
        return obj


class KerasAutoEncoder(KerasBaseEstimator, TransformerMixin):
    """
    Subclass of the KerasBaseEstimator to allow fitting to just X without requiring y.
    """

    def fit(self, X, y=None, **kwargs):
        if y is not None:
            logger.warning(
                f"This is an AutoEncoder and does not care about a "
                f"target, but a y was supplied. It will be ignored!"
            )
        y = X.copy()
        super().fit(X, y, **kwargs)
        return self

    def transform(self, X, **kwargs):

        with possible_tf_mgmt(self):
            xhat = self.model.predict(X, **kwargs)

        results = list()
        for sample_input, sample_output in zip(
            X.tolist(), xhat.reshape(X.shape).tolist()
        ):
            sample_input.extend(sample_output)
            results.append(sample_input)
        return np.asarray(results)

    def score(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> float:
        """
        Returns the explained variance score between auto encoder's input vs output

        Parameters
        ----------
        X: Union[np.ndarray, pd.DataFrame] - Input data to the model
        y: Union[np.ndarray, pd.DataFrame] - Target
        sample_weight: Optional[np.ndarray] - sample weights

        Returns
        -------
        score: float
        """
        if not hasattr(self, "model"):
            raise NotFittedError(
                f"This {self.__class__.__name__} has not been fitted yet."
            )

        with possible_tf_mgmt(self):
            out = self.model.predict(X)

        return explained_variance_score(X if y is None else y, out)
