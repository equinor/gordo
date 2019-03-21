# -*- coding: utf-8 -*-

import logging
import json
import math

from typing import Union, Callable, Dict, Any, Optional, Generator
from os import path
from contextlib import contextmanager
import pickle

import keras.models
import keras.backend as K
from keras.wrappers.scikit_learn import BaseWrapper
from keras.models import load_model
import numpy as np
import pandas as pd

from sklearn.base import TransformerMixin
from sklearn.metrics import explained_variance_score
from sklearn.exceptions import NotFittedError
from gordo_components.model.base import GordoBase


# This is required to run `register_model_builder` against registered factories
from gordo_components.model.factories import *  # pragma: no flakes

from gordo_components.model.register import register_model_builder


logger = logging.getLogger(__name__)


@contextmanager
def possible_tf_mgmt(keras_model):
    """
    Decorator - When serving a Keras model backed by tensorflow, the object is expected
    to have ``_tf_graph`` and ``_tf_session`` stored attrs. Which will be used as the
    default when calling the Keras model

    Notes
    -----
    Should only be needed by those developing new Keras models wrapped to behave like
    Scikit-Learn Esitmators

    Parameters
    ----------
    keras_model: KerasBaseEstimator
        An instance of a `KerasBaseEstimator` which can potentially have a tensorflow backend
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

        Parameters
        ----------
        kind: Union[callable, str]
            The structure of the model to build. As designated by any registered builder
            functions, registered with gordo_compontents.model.register.register_model_builder
            Alternatively, one may pass a builder function directly to this argument. Such a
            function should accept `n_features` as it's first argument, and pass any additional
            parameters to `**kwargs`

        kwargs: dict
            Any additional args which are passed to the factory
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
        """
        Parameters used for scikit learn kwargs"""
        return self.kwargs

    def fit(self, X: np.ndarray, y: Optional[np.ndarray], **kwargs):
        """
        Fit the model to X given y.

        Parameters
        ----------
        X: np.ndarray
            numpy array or pandas dataframe
        y: np.ndarray
            numpy array or pandas dataframe
        sample_weight: np.ndarray
            array like - weight to assign to samples
        kwargs
            Any additional kwargs to supply to keras fit method.

        Returns
        -------
        self
            'KerasAutoEncoder'
        """
        logger.debug(f"Fitting to data of length: {len(X)}")
        if len(X.shape) == 2:
            self.kwargs.update({"n_features": X.shape[1]})
        # for LSTM based models
        if len(X.shape) == 3:
            self.kwargs.update({"n_features": X.shape[2]})
        with possible_tf_mgmt(self):
            super().fit(X, y, sample_weight=None, **kwargs)
        return self

    def get_params(self, **params):
        """
        Gets the parameters for this estimator

        Parameters
        ----------
        params
            ignored (exists for API compatibility).

        Returns
        -------
        Dict[str, Any]
            Parameters used in this estimator
        """
        params = super().get_params(**params)
        params.update({"kind": self.kind})
        params.update(self.kwargs)
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
                if hasattr(self.model, "history") and self.model.history is not None:
                    f_name = path.join(directory, "history.pkl")
                    with open(f_name, "wb") as history_file:
                        pickle.dump(self.model.history, history_file)

    def score(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> float:
        """
        Returns the appropriate scoring metric for a given model.

        Parameters
        ----------
        X: Union[np.ndarray, pd.DataFrame]
            Input data to the model
        y: Union[np.ndarray, pd.DataFrame]
            Target
        sample_weight: Optional[np.ndarray]
            sample weights

        Returns
        -------
        score: float
            Returns chosen metric for the given model.
        """
        raise NotImplementedError(
            f"Subclasses of {self.__class__.__name__} must implement the .score(...) method."
        )

    @classmethod
    def load_from_dir(cls, directory: str):
        """
        Load an instance of this class from a directory, such that it was dumped to
        using :func:`gordo_components.model.models.KerasBaseEstimator.save_to_dir`

        Parameters
        ----------
        directory: str
            The directory to save this model to, must have write access

        Returns
        -------
        None
        """
        with open(path.join(directory, "params.json"), "r") as f:
            params = json.load(f)
        obj = cls(**params)
        model_file = path.join(directory, "model.h5")
        if path.isfile(model_file):
            with possible_tf_mgmt(obj):
                K.set_learning_phase(0)
                obj.model = load_model(model_file)
                history_file = path.join(directory, "history.pkl")
                if path.isfile(history_file):
                    with open(history_file, "rb") as hist_f:
                        obj.model.history = pickle.load(hist_f)

        return obj

    def get_metadata(self):
        """
        Get metadata for the KerasBaseEstimator.
        Includes a dictionary with key "history". The key's value is a a dictionary
        with a key "params" pointing another dictionary with various parameters.
        The metrics are defined in the params dictionary under "metrics".
        For each of the metrics there is a key who's value is a list of values for this
        metric per epoch.

        Returns
        -------
        Dict
            Metadata dictionary, including a history object if present
        """
        if (
            hasattr(self, "model")
            and hasattr(self.model, "history")
            and self.model.history
        ):
            history = self.model.history.history
            history["params"] = self.model.history.params
            return {"history": history}
        else:
            return {}


class KerasAutoEncoder(KerasBaseEstimator, TransformerMixin):
    """
    Subclass of the KerasBaseEstimator to allow fitting to just X without requiring y.
    """

    def fit(
        self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs
    ) -> "KerasAutoEncoder":
        if y is not None:
            logger.warning(
                f"This is an AutoEncoder and does not care about a "
                f"target, but a y was supplied. It will be ignored!"
            )
        y = X.copy()
        super().fit(X, y, **kwargs)
        return self

    def transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """

        Parameters
        ----------
        X: np.ndarray
            Input data
        kwargs: dict
            kwargs which are passed to Kera's ``predict`` method

        Notes
        -----
        The data returns from this method is double the feature length of its input.
        The first half of each sample output is the _input_ of the model, and the
        output is concatenated (axis=1) with the input. ie. If the input has 4 features,
        the output will have 8, where the first 4 are the values which went into the model.

        Returns
        -------
        np.ndarray

        """
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
        X: Union[np.ndarray, pd.DataFrame]
            Input data to the model
        y: Union[np.ndarray, pd.DataFrame]
            Target
        sample_weight: Optional[np.ndarray]
            sample weights

        Returns
        -------
        score: float
            Returns the explained variance score
        """
        if not hasattr(self, "model"):
            raise NotFittedError(
                f"This {self.__class__.__name__} has not been fitted yet."
            )

        with possible_tf_mgmt(self):
            out = self.model.predict(X)

        return explained_variance_score(X if y is None else y, out)


class KerasLSTMBaseEstimator(KerasBaseEstimator, TransformerMixin):
    """
       Subclass of the KerasBaseEstimator to allow to train a many-one LSTM autoencoder and an LSTM
       1 step forecast
    """

    def __init__(
        self,
        kind: Union[Callable, str],
        lookback_window: int = 1,
        batch_size: int = 32,
        **kwargs,
    ) -> None:
        """

        Parameters
        ----------
        kind: Union[Callable, str]
            The structure of the model to build. As designated by any registered builder
            functions, registered with gordo_components.model.register.register_model_builder
            Alternatively, one may pass a builder function directly to this argument. Such a
            function should accept `n_features` as it's first argument, and pass any additional
            parameters to `**kwargs`.
        lookback_window: int
            Number of timestamps (lags) used to train the model.
        batch_size: int
            Number of training examples used in one epoch.
        epochs: int
            Number of epochs to train the model. An epoch is an iteration over the entire
            data provided.
        verbose: int
            Verbosity mode. Possible values are 0, 1, or 2 where 0 = silent, 1 = progress bar,
            2 = one line per epoch.
        kwargs: dict
            Any arguments which are passed to the factory building function and/or any
            additional args to be passed to the intermediate fit method.
        """
        self.steps_per_epoch = None
        self.lookback_window = lookback_window
        self.batch_size = batch_size
        kwargs["lookback_window"] = lookback_window
        kwargs["kind"] = kind
        kwargs["batch_size"] = batch_size

        # fit_generator_params is a set of strings with the keyword arguments of Keras fit_generator method (excluding
        # "shuffle" as this will be hard coded).  This will be used in the fit method of the respective subclasses
        # to match the kwargs supplied when instantiating the subclass.
        # The matched kwargs will override the default kwargs of Keras fit_generator method when training the model.
        # Note: The decorator "@interfaces.legacy_generator_methods_support" to Keras' fit_generator method
        # does not forward any arguments to the inspect module
        self.fit_generator_params = {
            "steps_per_epoch",
            "epochs",
            "verbose",
            "callbacks",
            "validation_data",
            "validation_steps",
            "validation_freq",
            "class_weight",
            "max_queue_size",
            "workers",
            "use_multiprocessing",
            "initial_epoch",
        }
        super().__init__(**kwargs)

    def generate_window(
        self, X: np.ndarray, architecture: str, output_y: bool = True
    ) -> Generator[Union[np.ndarray, tuple], None, None]:
        """Returns the generator _generate_window.  Purpose of this function is to do the required checks
            prior to evaluate the generator.  If only _generate_window is used, ValueErrors may not be
            raised when implementing the transform method."""

        if architecture not in ["lstm_autoencoder", "lstm_forecast"]:
            raise ValueError(
                "invalid LSTM architecture. Choose either lstm_autoencoder or lstm_forecast "
            )

        if architecture == "lstm_autoencoder":
            if self.lookback_window > X.shape[0]:
                raise ValueError(
                    "For KerasLSTMAutoEncoder lookback_window must be <= size of X"
                )

        if architecture == "lstm_forecast":
            if self.lookback_window >= X.shape[0]:
                raise ValueError(
                    "For KerasLSTMForecast lookback_window must be < size of X"
                )

        return self._generate_window(X, architecture, output_y)

    def _generate_window(
        self, X: np.ndarray, architecture: str, output_y: bool = True
    ) -> Generator[Union[np.ndarray, tuple], None, None]:
        """

        Parameters:
        ----------
        X: 2D np array
            Data to predict/transform on (n_samples x n_features).
        output_y: bool
            If true, sample_in and sample_out for Keras LSTM fit_generator will be generated,
            If false, sample_in for Keras LSTM predict_generator will be generated.
        architecture: str
            "lstm_autoencoder" or "lstm_forecast"
            Type of LSTM architecture.
            If "lstm_autoencoder" then a many-one architecture will be
            implemented where the input will be the 3D array
            np.array([X[0 : lookback_window], X[1 : lookback_window+1],...])
            and the output is given by the 2D array:
            np.array([X[lookback_window-1], X[lookback_window],...])

            If "lstm_forecast" then a one-step forecast LSTM architecture will be
            implemented where the input will be the 3D array
            np.array([X[0 : lookback_window], X[1 : lookback_window],...])
            and the output is given by the 2D array:
            np.array([X[lookback_window], X[lookback_window+1],...])

        Returns:
        -------
        sample_in: 3D np array
            Each iterate generates a window of data points of size 1 x lookback_window
            x n_features to use within fit/transform methods.
        sample_out: 3D np array
            The last data point of sample_in (1 x 1 x n_features).

        """

        n_feat = X.shape[1]
        while True:
            if architecture == "lstm_autoencoder":
                for i in range(X.shape[0] - (self.lookback_window - 1)):
                    sample_in = X[i : self.lookback_window + i]
                    sample_out = sample_in[-1]
                    if output_y:
                        yield self._reshape_samples(n_feat, sample_in, sample_out)
                    else:
                        yield self._reshape_samples(n_feat, sample_in, sample_out)[0]
            if architecture == "lstm_forecast":
                for i in range(X.shape[0] - self.lookback_window):
                    sample_in = X[i : self.lookback_window + i]
                    sample_out = X[self.lookback_window + i, :]
                    if output_y:
                        yield self._reshape_samples(n_feat, sample_in, sample_out)
                    else:
                        yield self._reshape_samples(n_feat, sample_in, sample_out)[0]

    def _reshape_samples(self, n_feat, sample_in, sample_out):
        return (
            sample_in.reshape(1, self.lookback_window, n_feat),
            sample_out.reshape(1, n_feat),
        )

    def calc_steps_per_epoch(self, X):
        self.steps_per_epoch = math.ceil(
            (X.shape[0] - (self.lookback_window - 1)) / self.batch_size
        )
        return self.steps_per_epoch


class KerasLSTMAutoEncoder(KerasLSTMBaseEstimator):

    """
    Example
    -------
    >>> from gordo_components.model.factories.lstm_autoencoder import lstm_model
    >>> import numpy as np
    >>> from gordo_components.model.models import KerasLSTMAutoEncoder
    >>> lstm_ae = KerasLSTMAutoEncoder(kind="lstm_model",
    ...                                   lookback_window = 2,verbose=0)
    >>> X_train = np.random.random(size=300).reshape(100, 3)
    >>> model_fit = lstm_ae.fit(X_train)
    >>> X_test = np.random.random(size=12).reshape(4, 3)
    >>> model_transform = lstm_ae.transform(X_test)
    """

    def fit(
        self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs
    ) -> "KerasLSTMAutoEncoder":
        if y is not None:
            logger.warning(
                f"This is a many-to-one LSTM AutoEncoder that does not need a "
                f"target, but a y was supplied. It will not be used."
            )
        gen = self.generate_window(X, architecture="lstm_autoencoder", output_y=True)

        if not hasattr(self, "model"):
            # these are only used for the intermediate fit method (fit method of KerasBaseEstimator),
            # called only to initiate the model
            super().fit(*next(gen), epochs=1, verbose=0, **kwargs)

        steps_per_epoch = self.calc_steps_per_epoch(X)

        self.kwargs["steps_per_epoch"] = steps_per_epoch

        gen_kwargs = {
            k: v for k, v in self.kwargs.items() if k in self.fit_generator_params
        }

        with possible_tf_mgmt(self):
            # shuffle is set to False since we are dealing with time series data and
            # so training data will not be shuffled before each epoch.
            self.model.fit_generator(gen, shuffle=False, **gen_kwargs)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        gen = self.generate_window(X, architecture="lstm_autoencoder", output_y=False)
        with possible_tf_mgmt(self):
            xhat = self.model.predict_generator(
                gen, steps=X.shape[0] - (self.lookback_window - 1)
            )
        X = X[self.lookback_window - 1 :]
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
        (note: for LSTM X is offset by lookback_window).

        Parameters
        ----------
        X: Union[np.ndarray, pd.DataFrame]
            Input data to the model.
        y: Union[np.ndarray, pd.DataFrame]
            Target
        sample_weight: Optional[np.ndarray]
            Sample weights

        Returns
        -------
        score: float
            Returns the explained variance score.
        """
        if not hasattr(self, "model"):
            raise NotFittedError(
                f"This {self.__class__.__name__} has not been fitted yet."
            )

        out = self.transform(X)

        return explained_variance_score(out[:, : X.shape[1]], out[:, X.shape[1] :])


class KerasLSTMForecast(KerasLSTMBaseEstimator):

    """
    Example
    -------
    >>> from gordo_components.model.factories.lstm_autoencoder import lstm_hourglass
    >>> import numpy as np
    >>> from gordo_components.model.models import KerasLSTMForecast
    >>> lstm_ae = KerasLSTMForecast(kind="lstm_hourglass",
    ...                                   lookback_window = 2,verbose=0,architecture='lstm_forecast')
    >>> X_train = np.random.random(size=30).reshape(10, 3)
    >>> model_fit = lstm_ae.fit(X_train)
    """

    def fit(
        self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs
    ) -> "KerasLSTMForecast":
        if y is not None:
            logger.warning(
                f"This is a forecast based LSTM AutoEncoder that does not need a "
                f"target, but a y was supplied. It will not be used."
            )
        gen = self.generate_window(X, architecture="lstm_forecast", output_y=True)

        if not hasattr(self, "model"):
            # these are only used for the intermediate fit method (fit method of KerasBaseEstimator),
            # called only to initiate the model
            super().fit(*next(gen), epochs=1, verbose=0, **kwargs)

        steps_per_epoch = self.calc_steps_per_epoch(X)

        self.kwargs["steps_per_epoch"] = steps_per_epoch

        gen_kwargs = {
            k: v for k, v in self.kwargs.items() if k in self.fit_generator_params
        }

        with possible_tf_mgmt(self):
            # shuffle is set to False since we are dealing with time series data and
            # so training data will not be shuffled before each epoch.
            self.model.fit_generator(gen, shuffle=False, **gen_kwargs)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        gen = self.generate_window(X, architecture="lstm_forecast", output_y=False)
        with possible_tf_mgmt(self):
            xhat = self.model.predict_generator(
                gen, steps=X.shape[0] - self.lookback_window
            )
        X = X[self.lookback_window :]
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
        Returns the explained variance score between 1 step forecasted input and true input at next time step
        (note: for LSTM X is offset by lookback_window).

        Parameters
        ----------
        X: Union[np.ndarray, pd.DataFrame]
            Input data to the model.
        y: Union[np.ndarray, pd.DataFrame]
            Target
        sample_weight: Optional[np.ndarray]
            Sample weights

        Returns
        -------
        score: float
            Returns the explained variance score.
        """
        if not hasattr(self, "model"):
            raise NotFittedError(
                f"This {self.__class__.__name__} has not been fitted yet."
            )

        out = self.transform(X)

        return explained_variance_score(out[:, : X.shape[1]], out[:, X.shape[1] :])
