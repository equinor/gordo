# -*- coding: utf-8 -*-

import abc
import logging
import importlib
import tempfile
from pprint import pformat
from typing import Union, Callable, Dict, Any, Optional, Tuple
from abc import ABCMeta
from copy import copy, deepcopy
from importlib.util import find_spec

import tensorflow.keras.models
from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.preprocessing.sequence import pad_sequences, TimeseriesGenerator
from scikeras.wrappers import KerasRegressor
import numpy as np
import pandas as pd
import xarray as xr

from sklearn.base import TransformerMixin
from sklearn.metrics import explained_variance_score
from sklearn.exceptions import NotFittedError

from gordo import serializer
from gordo.machine.model.base import GordoBase

# This is required to run `register_model_builder` against registered factories
from gordo.machine.model.factories import *  # pragma: no flakes

from gordo.machine.model.register import register_model_builder

logger = logging.getLogger(__name__)


class KerasBaseEstimator(KerasRegressor, GordoBase):
    supported_fit_args = [
        "batch_size",
        "epochs",
        "verbose",
        "callbacks",
        "validation_split",
        "shuffle",
        "class_weight",
        "initial_epoch",
        "steps_per_epoch",
        "validation_batch_size",
        "max_queue_size",
        "workers",
        "use_multiprocessing",
    ]

    def __init__(
        self,
        kind: Union[
            str, Callable[[int, Dict[str, Any]], tensorflow.keras.models.Model]
        ],
        **kwargs,
    ) -> None:
        """
        Initialized a Scikit-Learn API compatitble Keras model with a pre-registered
        function or a builder function directly.

        Parameters
        ----------
        kind
            The structure of the model to build. As designated by any registered builder
            functions, registered with
            :func:`gordo.machine.model.register.register_model_builder`.
            Alternatively, one may pass a builder function directly to this argument.
            Such a function should accept `n_features` as it's first argument, and pass
            any additional parameters to `**kwargs`

        kwargs: dict
            Any additional args which are passed to the factory
            building function and/or any additional args to be passed
            to Keras' fit() method
        """
        self.kind = self.load_kind(kind)
        self.kwargs: Dict[str, Any] = kwargs
        self._history = None

        # This new keras wrapper expects most of these kwargs to be set to the model attributes and uses them for
        # defaults in some places, but always gives precedence to kwargs passed to respective fit, predict and compile
        # methods, so this is just to make it happy again
        _expected_kwargs = {
            *KerasRegressor._fit_kwargs,
            *KerasRegressor._predict_kwargs,
            *KerasRegressor._compile_kwargs,
        }
        KerasRegressor.__init__(
            self,
            **{key: value for key, value in kwargs.items() if key in _expected_kwargs},
        )

    @staticmethod
    def parse_module_path(module_path) -> Tuple[Optional[str], str]:
        module_paths = module_path.split(".")
        if len(module_paths) == 1:
            return None, module_paths[0]
        else:
            return ".".join(module_paths[:-1]), module_paths[-1]

    def load_kind(self, kind):
        if callable(kind):
            register_model_builder(type=self.__class__.__name__)(kind)
            return kind.__name__
        else:
            module_name, class_name = self.parse_module_path(kind)
            if module_name is None:
                if (
                    class_name
                    not in register_model_builder.factories[self.__class__.__name__]
                ):
                    raise ValueError(
                        f"kind: {kind} is not an available model for type: {class_name}!"
                    )
            else:
                has_error = True
                try:
                    has_error = not find_spec(module_name)
                except ModuleNotFoundError:
                    pass
                if has_error:
                    raise ValueError(
                        f"kind: {kind}, unable to find module: '{module_name}'"
                    )
            return kind

    @classmethod
    def extract_supported_fit_args(cls, kwargs):
        """
        Filtering only ``fit`` related kwargs

        Parameters
        ----------
        kwargs

        """
        fit_args = {}
        for arg in cls.supported_fit_args:
            if arg in kwargs:
                fit_args[arg] = kwargs[arg]
        return fit_args

    @classmethod
    def from_definition(cls, definition: dict):
        """
        Handler for :func:`gordo.serializer.from_definition`

        Parameters
        ----------
        definition
            Model definition

        """
        kind = definition.pop("kind")
        kwargs = copy(definition)
        return cls(kind, **kwargs)

    def into_definition(self) -> dict:
        """
        Handler for ``gordo.serializer.into_definition``

        Returns
        -------

        """
        definition = copy(self.kwargs)
        definition["kind"] = self.kind
        return definition

    @property
    def sk_params(self):
        """
        Parameters used for scikit learn kwargs"""
        fit_args = self.extract_supported_fit_args(self.kwargs)
        if fit_args:
            kwargs = deepcopy(self.kwargs)
            kwargs.update(serializer.load_params_from_definition(fit_args))
            return kwargs
        else:
            return self.kwargs

    def __getstate__(self):

        state = self.__dict__.copy()

        if self.model is not None:
            with tempfile.NamedTemporaryFile("w", suffix=".keras") as tf:
                save_model(self.model, tf.name, overwrite=True)
                with open(tf.name, "rb") as inf:
                    state["model"] = inf.read()

            from tensorflow.python.keras.callbacks import History

            history = History()
            history.history = self._history.history
            history.params = self._history.params
            history.epoch = self._history.epoch
            state["history"] = history
        return state

    def __setstate__(self, state):
        if "model" in state and state["model"] is not None:
            with tempfile.NamedTemporaryFile("wb", suffix=".keras") as tf:
                tf.write(state["model"])
                state["model"] = load_model(tf.name, compile=False)
        self.__dict__ = state
        return self

    @staticmethod
    def get_n_features_out(
        y: Union[np.ndarray, pd.DataFrame, xr.DataArray]
    ) -> Union[int, tuple]:
        shape_len = len(y.shape)
        if shape_len == 1:
            raise ValueError(
                "Unsupported number of the output dataset dimensions %d" % shape_len
            )
        elif shape_len == 2:
            return y.shape[1]
        else:
            return y.shape[1:]

    @staticmethod
    def get_n_features(
        X: Union[np.ndarray, pd.DataFrame, xr.DataArray]
    ) -> Union[int, tuple]:
        shape_len = len(X.shape)
        if shape_len == 1:
            raise ValueError(
                "Unsupported number of the output dataset dimensions %d" % shape_len
            )
        elif shape_len == 2:
            return X.shape[1]
        else:
            # TODO fix for the legacy LSTM
            if not isinstance(X, xr.DataArray):
                return X.shape[2]
            return X.shape[1:]

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame, xr.DataArray],
        y: Union[np.ndarray, pd.DataFrame, xr.DataArray],
        **kwargs,
    ):
        """
        Fit the model to X given y.

        Parameters
        ----------
        X
            numpy array or pandas dataframe
        y
            numpy array or pandas dataframe
        sample_weight
            array like - weight to assign to samples
        kwargs
            Any additional kwargs to supply to keras fit method.
        """

        # Reshape y if needed, and set n features of target
        if isinstance(y, np.ndarray) and y.ndim == 1:
            y = y.reshape(-1, 1)

        logger.debug(f"Fitting to data of length: {len(X)}")
        self.kwargs.update(
            {
                "n_features": self.get_n_features(X),
                "n_features_out": self.get_n_features_out(y),
            }
        )

        if isinstance(X, (pd.DataFrame, xr.DataArray)):
            X = X.values
        if isinstance(y, (pd.DataFrame, xr.DataArray)):
            y = y.values
        kwargs.setdefault("verbose", 0)

        if self.model is None:
            self._prepare_model()
        model = super().fit(X, y, sample_weight=None, **kwargs)
        if isinstance(model, KerasRegressor):
            self._history = model.model.history
        return self

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """

        Parameters
        ----------
        X
            Input data
        kwargs
            kwargs which are passed to Kera's ``predict`` method
        """
        kwargs.setdefault("verbose", 0)
        return self.model.predict(X, **kwargs)

    def get_params(self, **params):
        """
        Gets the parameters for this estimator

        Parameters
        ----------
        params
            ignored (exists for API compatibility).

        Returns
        -------
            Parameters used in this estimator
        """
        params = super().get_params(**params)
        params.update({"kind": self.kind})
        params.update(self.kwargs)
        if self.kwargs.get("callbacks") is not None and any(
            isinstance(callback, dict) for callback in self.kwargs["callbacks"]
        ):
            params["callbacks"] = serializer.build_callbacks(self.kwargs["callbacks"])
        return params

    def _prepare_model(self):
        module_name, class_name = self.parse_module_path(self.kind)
        if module_name is None:
            factories = register_model_builder.factories[self.__class__.__name__]
            model = factories[self.kind]
        else:
            module = importlib.import_module(module_name)
            if not hasattr(module, class_name):
                raise ValueError(
                    "kind: %s, unable to find class %s in module '%s'"
                    % (self.kind, class_name, module_name)
                )
            model = getattr(module, class_name)
        self.model = model(**self.sk_params)

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
            Metadata dictionary, including a history object if present
        """
        if self._history is not None:
            history = self._history.history
            history["params"] = self._history.params
            return {"history": history}
        else:
            return {}


class KerasAutoEncoder(KerasBaseEstimator, TransformerMixin):
    """
    Subclass of the KerasBaseEstimator to allow fitting to just X without requiring y.
    """

    def score(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.DataFrame],
        sample_weight: Optional[np.ndarray] = None,
        **kwargs,
    ) -> float:
        """
        Returns the explained variance score between auto encoder's input vs output

        Parameters
        ----------
        X
            Input data to the model
        y
            Target
        sample_weight
            sample weights
        kwargs
            Additional kwargs for model.predict()

        Returns
        -------
            Returns the explained variance score
        """
        if self.model is None:
            raise NotFittedError(
                f"This {self.__class__.__name__} has not been fitted yet."
            )

        kwargs.setdefault("verbose", 0)
        out = self.model.predict(X, **kwargs)

        return explained_variance_score(y, out)


class KerasRawModelRegressor(KerasAutoEncoder):
    """
    Create a scikit-learn like model with an underlying ``tensorflow.keras`` model
    from a raw config.

    Examples
    --------
    >>> import yaml
    >>> import numpy as np
    >>> config_str = '''
    ...   # Arguments to the .compile() method
    ...   compile:
    ...     loss: mse
    ...     optimizer: adam
    ...
    ...   # The architecture of the model itself.
    ...   spec:
    ...     tensorflow.keras.models.Sequential:
    ...       layers:
    ...         - tensorflow.keras.layers.Dense:
    ...             units: 4
    ...             input_shape: [4]
    ...         - tensorflow.keras.layers.Dense:
    ...             units: 1
    ... '''
    >>> config = yaml.safe_load(config_str)
    >>> model = KerasRawModelRegressor(kind=config)
    >>>
    >>> X, y = np.random.random((10, 4)), np.random.random((10, 1))
    >>> model.fit(X, y, verbose=0)
    KerasRawModelRegressor(kind: {'compile': {'loss': 'mse', 'optimizer': 'adam'},
     'spec': {'tensorflow.keras.models.Sequential': {'layers': [{'tensorflow.keras.layers.Dense': {'input_shape': [4],
                                                                                                   'units': 4}},
                                                                {'tensorflow.keras.layers.Dense': {'units': 1}}]}}})
    >>> out = model.predict(X)
    """

    _expected_keys = ("spec", "compile")

    def load_kind(self, kind):
        return kind

    def __repr__(self):
        return f"{self.__class__.__name__}(kind: {pformat(self.kind)})"

    def _prepare_model(self):
        """Build Keras model from specification"""
        if not all(k in self.kind for k in self._expected_keys):
            raise ValueError(
                f"Expected spec to have keys: {self._expected_keys}, but found {self.kind.keys()}"
            )
        logger.debug(f"Building model from spec: {self.kind}")

        model = serializer.from_definition(self.kind["spec"])

        # Load any compile kwargs as well, such as compile.optimizer which may map to class obj
        kwargs = serializer.from_definition(self.kind["compile"])
        model.compile(**kwargs)

        self.model = model


class KerasLSTMBaseEstimator(KerasBaseEstimator, TransformerMixin, metaclass=ABCMeta):
    """
    Abstract Base Class to allow to train a many-one LSTM autoencoder and an LSTM 1 step forecast
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
        kind
            The structure of the model to build. As designated by any registered builder
            functions, registered with :func:`gordo.machine.model.register.register_model_builder`.
            Alternatively, one may pass a builder function directly to this argument.
            Such a function should accept ``n_features`` as it's first argument, and pass
            any additional parameters to ``**kwargs``.
        lookback_window
            Number of timestamps (lags) used to train the model.
        batch_size
            Number of training examples used in one epoch.
        epochs
            Number of epochs to train the model. An epoch is an iteration over the
            entire data provided.
        verbose
            Verbosity mode. Possible values are 0, 1, or 2 where 0 = silent,
            1 = progress bar, 2 = one line per epoch.
        kwargs
            Any arguments which are passed to the factory building function and/or any
            additional args to be passed to the intermediate fit method.
        """
        self.lookback_window = lookback_window
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

    @abc.abstractproperty
    def lookahead(self) -> int:
        """Steps ahead in y the model should target"""
        ...

    def get_metadata(self):
        """
        Add number of forecast steps to metadata

        Returns
        -------
            Metadata dictionary, including forecast steps.
        """
        metadata = super().get_metadata()
        metadata.update({"forecast_steps": self.lookahead})
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

    def fit(  # type: ignore
        self, X: np.ndarray, y: np.ndarray, **kwargs
    ) -> "KerasLSTMForecast":
        """
        This fits a one step forecast LSTM architecture.

        Parameters
        ----------
        X
           2D numpy array of dimension n_samples x n_features. Input data to train.
        y
           2D numpy array representing the target
        kwargs
            Any additional args to be passed to Keras ``fit_generator`` method.

        Returns
        -------
            KerasLSTMForecast

        """

        X = X.values if isinstance(X, pd.DataFrame) else X
        y = y.values if isinstance(y, pd.DataFrame) else y

        X = self._validate_and_fix_size_of_X(X)

        # We call super.fit on a single sample (notice the batch_size=1) to initiate the
        # model using the scikit-learn wrapper.
        tsg = create_keras_timeseriesgenerator(
            X=X[
                : self.lookahead + self.lookback_window
            ],  # We only need a bit of the data
            y=y[: self.lookahead + self.lookback_window],
            batch_size=1,
            lookback_window=self.lookback_window,
            lookahead=self.lookahead,
        )

        primer_x, primer_y = tsg[0]

        super().fit(X=primer_x, y=primer_y, epochs=1, verbose=0)

        tsg = create_keras_timeseriesgenerator(
            X=X,
            y=y,
            batch_size=self.batch_size,
            lookback_window=self.lookback_window,
            lookahead=self.lookahead,
        )

        gen_kwargs = {
            k: v
            for k, v in {**self.kwargs, **kwargs}.items()
            if k in self.fit_generator_params
        }

        # shuffle is set to False since we are dealing with time series data and
        # so training data will not be shuffled before each epoch.
        self.model.fit(tsg, shuffle=False, **gen_kwargs)
        return self

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Parameters
        ----------
        X
            Data to predict/transform. 2D numpy array of dimension ``n_samples x n_features`` where ``n_samples`` must be > lookback_window.

        Returns
        -------
            2D numpy array of dimension ``(n_samples - lookback_window) x 2*n_features``.
            The first half of the array ``(results[:, :n_features])`` corresponds to X offset
            by ``lookback_window+1`` (i.e., ``X[lookback_window:,:]``) whereas the second half corresponds to
            the predicted values of ``X[lookback_window:,:]``.

        Example
        -------
        >>> import numpy as np
        >>> from gordo.machine.model.factories.lstm_autoencoder import lstm_model
        >>> from gordo.machine.model.models import KerasLSTMForecast
        >>> #Define train/test data
        >>> X_train = np.array([[1, 1], [2, 3], [0.5, 0.6], [0.3, 1], [0.6, 0.7]])
        >>> X_test = np.array([[2, 3], [1, 1], [0.1, 1], [0.5, 2]])
        >>> #Initiate model, fit and transform
        >>> lstm_ae = KerasLSTMForecast(kind="lstm_model",
        ...                             lookback_window=2,
        ...                             verbose=0)
        >>> model_fit = lstm_ae.fit(X_train, y=X_train.copy())
        >>> model_transform = lstm_ae.predict(X_test)
        >>> model_transform.shape
        (2, 2)
        """
        X = X.values if isinstance(X, pd.DataFrame) else X

        X = self._validate_and_fix_size_of_X(X)
        tsg = create_keras_timeseriesgenerator(
            X=X,
            y=X,
            batch_size=10000,
            lookback_window=self.lookback_window,
            lookahead=self.lookahead,
        )
        kwargs.setdefault("verbose", 0)
        return self.model.predict(tsg, **kwargs)

    def score(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.DataFrame],
        sample_weight: Optional[np.ndarray] = None,
        **kwargs,
    ) -> float:
        """
        Returns the explained variance score between 1 step forecasted input and true
        input at next time step (note: for LSTM X is offset by ``lookback_window``).

        Parameters
        ----------
        X
            Input data to the model.
        y
            Target
        sample_weight
            Sample weights
        kwargs
            Additional kwargs for predict

        Returns
        -------
            Returns the explained variance score.
        """
        if self.model is None:
            raise NotFittedError(
                f"This {self.__class__.__name__} has not been fitted yet."
            )

        kwargs.setdefault("verbose", 0)
        out = self.predict(X, **kwargs)

        # Limit X samples to match the offset causes by LSTM lookback window
        # ie, if look back window is 5, 'out' will be 5 rows less than X by now
        return explained_variance_score(y[-len(out) :], out)


class KerasLSTMForecast(KerasLSTMBaseEstimator):
    @property
    def lookahead(self) -> int:
        return 1


class KerasLSTMAutoEncoder(KerasLSTMBaseEstimator):
    @property
    def lookahead(self) -> int:
        return 0


def create_keras_timeseriesgenerator(
    X: np.ndarray,
    y: Optional[np.ndarray],
    batch_size: int,
    lookback_window: int,
    lookahead: int,
) -> tensorflow.keras.preprocessing.sequence.TimeseriesGenerator:
    """
    Provides a :class:`keras.preprocessing.sequence.TimeseriesGenerator` for use with
    LSTM's, but with the added ability to specify the lookahead of the target in y.

    If lookahead==0 then the generated samples in X will have as their last element
    the same as the corresponding Y. If lookahead is 1 then the values in Y is shifted
    so it is one step in the future compared to the last value in the samples in X,
    and similar for larger values.


    Parameters
    ----------
    X
        2d array of values, each row being one sample.
    y
        array representing the target.
    batch_size
        How big should the generated batches be?
    lookback_window
        How far back should each sample see. 1 means that it contains a single
        measurement
    lookahead
        How much is Y shifted relative to X

    Returns
    -------
        3d matrix with a list of batchX-batchY pairs, where batchX is a batch of
        X-values, and correspondingly for batchY. A batch consist of ``batch_size`` nr
        of pairs of samples (or y-values), and each sample is a list of length
        ``lookback_window``.

    Examples
    -------
    >>> import numpy as np
    >>> X, y = np.random.rand(100,2), np.random.rand(100, 2)
    >>> gen = create_keras_timeseriesgenerator(X, y,
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
    new_length = len(X) + 1 - lookahead
    kwargs: Dict[str, Any] = dict(length=lookback_window, batch_size=batch_size)
    if lookahead == 1:
        kwargs.update(dict(data=X, targets=y))

    elif lookahead >= 0:

        pad_kw = dict(maxlen=new_length, dtype=X.dtype)

        if lookahead == 0:
            kwargs["data"] = pad_sequences([X], padding="post", **pad_kw)[0]
            kwargs["targets"] = pad_sequences([y], padding="pre", **pad_kw)[0]

        elif lookahead > 1:
            kwargs["data"] = pad_sequences(
                [X], padding="post", truncating="post", **pad_kw
            )[0]
            kwargs["targets"] = pad_sequences(
                [y], padding="pre", truncating="pre", **pad_kw
            )[0]
    else:
        raise ValueError(f"Value of `lookahead` can not be negative, is {lookahead}")

    return TimeseriesGenerator(**kwargs)
