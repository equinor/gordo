# -*- coding: utf-8 -*-

import logging
import json

from typing import Union, Callable, Dict, Any, Optional
from os import path
from contextlib import contextmanager
import pickle
from abc import ABCMeta

import keras.models
import keras.backend as K
from keras.preprocessing.sequence import pad_sequences, TimeseriesGenerator
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
        An instance of a `KerasBaseEstimator` which can potentially have a tensorflow
        backend
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
        Initialized a Scikit-Learn API compatitble Keras model with a pre-registered
        function or a builder function
        directly.

        Parameters
        ----------
        kind: Union[callable, str]
            The structure of the model to build. As designated by any registered builder
            functions, registered with
            `gordo_compontents.model.register.register_model_builder`.
            Alternatively, one may pass a builder function directly to this argument.
            Such a function should accept `n_features` as it's first argument, and pass
            any additional parameters to `**kwargs`

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
        params.pop("build_fn", None)
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

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """

        Parameters
        ----------
        X: np.ndarray
            Input data
        kwargs: dict
            kwargs which are passed to Kera's ``predict`` method


        Returns
        -------
        results:
            np.ndarray
        """
        with possible_tf_mgmt(self):
            xhat = self.model.predict(X, **kwargs)
        return xhat

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

        return explained_variance_score(X, out)


class KerasLSTMBaseEstimator(KerasBaseEstimator, TransformerMixin, metaclass=ABCMeta):
    """
    Abstract Base Class to allow to train a many-one LSTM autoencoder and an LSTM
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
            functions, registered with
            `gordo_components.model.register.register_model_builder`.
            Alternatively, one may pass a builder function directly to this argument.
            Such a function should accept `n_features` as it's first argument, and pass
            any additional parameters to `**kwargs`.
        lookback_window: int
            Number of timestamps (lags) used to train the model.
        batch_size: int
            Number of training examples used in one epoch.
        epochs: int
            Number of epochs to train the model. An epoch is an iteration over the
            entire data provided.
        verbose: int
            Verbosity mode. Possible values are 0, 1, or 2 where 0 = silent,
            1 = progress bar, 2 = one line per epoch.
        kwargs: dict
            Any arguments which are passed to the factory building function and/or any
            additional args to be passed to the intermediate fit method.
        """
        self.lookback_window = lookback_window
        self.batch_size = batch_size
        kwargs["lookback_window"] = lookback_window
        kwargs["kind"] = kind
        kwargs["batch_size"] = batch_size

        # fit_generator_params is a set of strings with the keyword arguments of
        # Keras fit_generator method (excluding "shuffle" as this will be hardcoded).
        # This will be used in the fit method of the respective subclasses to match
        # the kwargs supplied when instantiating the subclass. The matched kwargs
        # will override the default kwargs of Keras fit_generator method when
        # training the model. Note: The decorator
        # "@interfaces.legacy_generator_methods_support" to Keras' fit_generator
        # method does not forward any arguments to the inspect module
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


class KerasLSTMAutoEncoder(KerasLSTMBaseEstimator):
    def _validate_and_fix_size_of_X(self, X):
        if X.ndim == 1:
            logger.info(
                f"Reshaping X from an array to an matrix of shape {(len(X), 1)}"
            )
            X = X.reshape(len(X), 1)

        if self.lookback_window > X.shape[0]:
            raise ValueError(
                "For KerasLSTMAutoEncoder lookback_window must be <= size of X"
            )
        return X

    def fit(
        self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs
    ) -> "KerasLSTMAutoEncoder":
        """
        This fits a many-to-one LSTM architecture using Keras fit_generator method.

        Parameters
        ----------
        X: np.ndarray
           2D numpy array of dimension n_samples x n_features. Input data to train.
        y: Optional[np.ndarray]
           This is an LSTM Autoencoder which does not need a y. Thus it will be ignored,
           and is only added as a parameter to be compatible with scikit-learn.
        kwargs: dict
            Any additional args to be passed to Keras fit_generator method.

        Returns
        -------
        class:
            KerasLSTMAutoEncoder

        """

        if y is not None:
            logger.warning(
                f"This is a many-to-one LSTM AutoEncoder that does not need a "
                f"target, but a y was supplied. It will not be used."
            )

        X = self._validate_and_fix_size_of_X(X)

        # We call super.fit on a single sample (notice the batch_size=1) to initiate the
        # model using the scikit-learn wrapper.
        lookahead = 0
        tsg = create_keras_timeseriesgenerator(
            data=X[
                : lookahead + self.lookback_window
            ],  # We only need a bit of the data
            batch_size=1,
            lookback_window=self.lookback_window,
            lookahead=lookahead,
        )
        x, y = tsg[0]
        super().fit(X=x, y=y, epochs=1, verbose=0)

        tsg = create_keras_timeseriesgenerator(
            data=X,
            batch_size=self.batch_size,
            lookback_window=self.lookback_window,
            lookahead=0,
        )
        gen_kwargs = {
            k: v
            for k, v in {**self.kwargs, **kwargs}.items()
            if k in self.fit_generator_params
        }
        with possible_tf_mgmt(self):
            # shuffle is set to False since we are dealing with time series data and
            # so training data will not be shuffled before each epoch.
            self.model.fit_generator(tsg, shuffle=False, **gen_kwargs)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """

        Parameters
        ----------
         X: np.ndarray
            Data to autoencode. 2D numpy array of dimension n_samples x n_features where
            n_samples must be >= lookback_window.


        Returns
        -------
        results: np.ndarray
            2D numpy array of dimension `(n_samples - (lookback_window - 1)) x
            2*n_features`.  The first half of the array ( results[:,:n_features])
            corresponds to X offset by the lookback_window (i.e., X[lookback_window -
            1:, :]) whereas the second half corresponds to the autoencoded values of
            X[ lookback_window - 1:,:].


        Example
        -------
        >>> import numpy as np
        >>> import tensorflow as tf
        >>> import random as rn
        >>> from keras import backend as K
        >>> from gordo_components.model.factories.lstm_autoencoder import lstm_model
        >>> from gordo_components.model.models import KerasLSTMAutoEncoder
        >>> #Setting seeds to get reproducible results
        >>> session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
        ...                               inter_op_parallelism_threads=1)
        >>> sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        >>> K.set_session(sess)
        >>> np.random.seed(42)
        >>> rn.seed(12345)
        >>> tf.set_random_seed(1234)
        >>> #Define train/test data
        >>> X_train = np.array([[1, 1], [2, 3], [0.5, 0.6], [0.3, 1], [0.6, 0.7]])
        >>> X_test = np.array([[2, 3], [1, 1], [0.1, 1], [0.5, 2]])
        >>> #Initiate model, fit and transform
        >>> lstm_ae = KerasLSTMAutoEncoder(kind="lstm_model",
        ...                                lookback_window=2,
        ...                                verbose=0)
        >>> model_fit = lstm_ae.fit(X_train)
        >>> model_transform = lstm_ae.predict(X_test)
        >>> model_transform.shape
        (3, 2)
        """
        X = self._validate_and_fix_size_of_X(X)
        # predict batch_size does not need to correspond to
        # training batch_size
        tsg = create_keras_timeseriesgenerator(
            data=X, batch_size=10000, lookback_window=self.lookback_window, lookahead=0
        )
        with possible_tf_mgmt(self):
            return self.model.predict_generator(tsg)

    def score(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> float:
        """
        Returns the explained variance score between auto encoder's input vs output
        (note: for LSTM X is offset by lookback_window - 1).

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

        out = self.predict(X)

        # Limit X samples to match the offset causes by LSTM lookback window
        # ie, if look back window is 5, 'out' will be 5 rows less than X by now
        return explained_variance_score(X[-len(out) :], out)


class KerasLSTMForecast(KerasLSTMBaseEstimator):
    def get_metadata(self):
        """
        Add number of forecast steps to metadata

        Returns
        -------
        metadata: dict
            Metadata dictionary, including forecast steps.
        """
        forecast_steps = {"forecast_steps": 1}
        metadata = super().get_metadata()
        metadata.update(forecast_steps)
        return metadata

    def _validate_and_fix_size_of_X(self, X):
        if X.ndim == 1:
            logger.info(
                f"Reshaping X from an array to an matrix of shape {(len(X), 1)}"
            )
            X = X.reshape(len(X), 1)

        if self.lookback_window >= X.shape[0]:
            raise ValueError(
                "For KerasLSTMForecast lookback_window must be < size of X"
            )
        return X

    def fit(
        self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs
    ) -> "KerasLSTMForecast":

        """
        This fits a one step forecast LSTM architecture.

        Parameters
        ----------
        X: np.ndarray
           2D numpy array of dimension n_samples x n_features. Input data to train.
        y: Optional[np.ndarray]
           This is an LSTM Autoencoder which does not need a y. Thus it will be ignored,
           and is only added as a parameter to be compatible with scikit-learn.
        kwargs: dict
            Any additional args to be passed to Keras `fit_generator` method.

        Returns
        -------
        class:
            KerasLSTMForecast

        """
        if y is not None:
            logger.warning(
                f"This is a forecast based LSTM model that does not need a "
                f"target, but a y was supplied. It will not be used."
            )

        X = self._validate_and_fix_size_of_X(X)

        # We call super.fit on a single sample (notice the batch_size=1) to initiate the
        # model using the scikit-learn wrapper.
        lookahead = 1
        tsg = create_keras_timeseriesgenerator(
            data=X[
                : lookahead + self.lookback_window
            ],  # We only need a bit of the data
            batch_size=1,
            lookback_window=self.lookback_window,
            lookahead=lookahead,
        )
        x, y = tsg[0]
        super().fit(X=x, y=y, epochs=1, verbose=0)

        tsg = create_keras_timeseriesgenerator(
            data=X,
            batch_size=self.batch_size,
            lookback_window=self.lookback_window,
            lookahead=1,
        )

        gen_kwargs = {
            k: v
            for k, v in {**self.kwargs, **kwargs}.items()
            if k in self.fit_generator_params
        }

        with possible_tf_mgmt(self):
            # shuffle is set to False since we are dealing with time series data and
            # so training data will not be shuffled before each epoch.
            self.model.fit_generator(tsg, shuffle=False, **gen_kwargs)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
         X: np.ndarray
            Data to predict/transform. 2D numpy array of dimension `n_samples x
            n_features` where `n_samples` must be > lookback_window.

        Returns
        -------
        results: np.ndarray
                 2D numpy array of dimension `(n_samples - lookback_window) x
                 2*n_features`.  The first half of the array `(results[:,
                 :n_features])` corresponds to X offset by `lookback_window+1` (i.e.,
                 `X[lookback_window:,:]`) whereas the second half corresponds to the
                 predicted values of `X[lookback_window:,:]`.


        Example
        -------
        >>> import numpy as np
        >>> import tensorflow as tf
        >>> import random as rn
        >>> from keras import backend as K
        >>> from gordo_components.model.factories.lstm_autoencoder import lstm_model
        >>> from gordo_components.model.models import KerasLSTMForecast
        >>> #Setting seeds to get reproducible results
        >>> session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
        ...                               inter_op_parallelism_threads=1)
        >>> sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        >>> K.set_session(sess)
        >>> np.random.seed(42)
        >>> rn.seed(12345)
        >>> tf.set_random_seed(1234)
        >>> #Define train/test data
        >>> X_train = np.array([[1, 1], [2, 3], [0.5, 0.6], [0.3, 1], [0.6, 0.7]])
        >>> X_test = np.array([[2, 3], [1, 1], [0.1, 1], [0.5, 2]])
        >>> #Initiate model, fit and transform
        >>> lstm_ae = KerasLSTMForecast(kind="lstm_model",
        ...                             lookback_window=2,
        ...                             verbose=0)
        >>> model_fit = lstm_ae.fit(X_train)
        >>> model_transform = lstm_ae.predict(X_test)
        >>> model_transform.shape
        (2, 2)
        """

        X = self._validate_and_fix_size_of_X(X)
        tsg = create_keras_timeseriesgenerator(
            data=X, batch_size=10000, lookback_window=self.lookback_window, lookahead=1
        )
        with possible_tf_mgmt(self):
            return self.model.predict_generator(tsg)

    def score(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> float:
        """
        Returns the explained variance score between 1 step forecasted input and true
        input at next time step (note: for LSTM X is offset by `lookback_window`).

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

        out = self.predict(X)

        # Limit X samples to match the offset causes by LSTM lookback window
        # ie, if look back window is 5, 'out' will be 5 rows less than X by now
        return explained_variance_score(X[-len(out) :], out)


def create_keras_timeseriesgenerator(
    data: np.ndarray, batch_size: int, lookback_window: int, lookahead: int
) -> keras.preprocessing.sequence.TimeseriesGenerator:
    """
    Provides a `keras.preprocessing.sequence.TimeseriesGenerator` for use with
    LSTM's, but with the added ability to specify the lookahead of the target in y.

    If lookahead==0 then the generated samples in X will have as their last element
    the same as the corresponding Y. If lookahead is 1 then the values in Y is shifted
    so it is one step in the future compared to the last value in the samples in X,
    and similar for larger values.


    Parameters
    ----------
    data: np.ndarray
        2d array of values, each row being one sample.
    batch_size: int
        How big should the generated batches be?
    lookback_window: int
        How far back should each sample see. 1 means that it contains a single
        measurement
    lookahead: int
        How much is Y shifted relative to X

    Returns
    -------
    TimeseriesGenerator
        3d matrix with a list of batchX-batchY pairs, where batchX is a batch of
        X-values, and correspondingly for batchY. A batch consist of `batch_size` nr
        of pairs of samples (or y-values), and each sample is a list of length
        `lookback_window`.

    Examples
    -------
    >>> import numpy as np
    >>> data = np.random.rand(100,2)
    >>> gen = create_keras_timeseriesgenerator(data,
    ...                                        batch_size=10,
    ...                                        lookback_window=20,
    ...                                        lookahead=0)
    >>> len(gen) # 9 = (100-20+1)/10
    9
    >>> len(gen[0]) # batchX and batchY
    2
    >>> len(gen[0][0]) # batch_size=10
    10
    >>> len(gen[0][0][0]) # a single sample, lookback_window = 20,
    20
    >>> len(gen[0][0][0][0]) # n_features = 2
    2
    """
    new_length = len(data) + 1 - lookahead

    if lookahead == 1:
        input_data = data
        target_data = data
    elif lookahead == 0:
        input_data = pad_sequences(
            [data], maxlen=new_length, padding="post", dtype=data.dtype
        )[0]
        target_data = pad_sequences(
            [data], maxlen=new_length, padding="pre", dtype=data.dtype
        )[0]

    elif lookahead > 1:
        input_data = pad_sequences(
            [data],
            maxlen=new_length,
            padding="post",
            dtype=data.dtype,
            truncating="post",
        )[0]
        target_data = pad_sequences(
            [data], maxlen=new_length, padding="pre", dtype=data.dtype, truncating="pre"
        )[0]
    else:
        raise ValueError(f"Value of `lookahead` can not be negative, is {lookahead}")

    tsgen = TimeseriesGenerator(
        input_data, targets=target_data, length=lookback_window, batch_size=batch_size
    )
    return tsgen
