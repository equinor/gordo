# -*- coding: utf-8 -*-

from typing import Tuple, Union, Dict, Any

import keras.optimizers
from keras.optimizers import Optimizer
from keras.layers import Dense, LSTM
from keras.models import Sequential as KerasSequential

from gordo_components.model.register import register_model_builder
from gordo_components.model.factories.model_factories_utils import hourglass_calc_dims


@register_model_builder(type="KerasLSTMAutoEncoder")
@register_model_builder(type="KerasLSTMForecast")
def lstm_model(
    n_features: int,
    lookback_window: int = 1,
    encoding_dim: Tuple[int, ...] = (256, 128, 64),
    decoding_dim: Tuple[int, ...] = (64, 128, 256),
    encoding_func: Tuple[str, ...] = ("tanh", "tanh", "tanh"),
    decoding_func: Tuple[str, ...] = ("tanh", "tanh", "tanh"),
    out_func: str = "linear",
    optimizer: Union[str, Optimizer] = "adam",
    optimizer_kwargs: Dict[str, Any] = dict(),
    compile_kwargs: Dict[str, Any] = dict(),
    **kwargs,
) -> keras.models.Sequential:
    """
    Builds a customized Keras LSTM neural network auto-encoder based on a config dict.

    Parameters
    ----------
    n_features: int
        Number of features the dataset X will contain.
    lookback_window: int
        Number of timesteps used to train the model.
        One timestep = current observation in the sample.
        Two timesteps = current observation + previous observation in the sample.
        ...
    encoding_func: list
        Activation functions for the encoder part.
    decoding_func: list
        Activation functions for the decoder part.
    func_output: str
        Activation function for the output Dense layer.
    enc_dim: list
        List of numbers with the number of neurons in the encoding part.
    dec_dim: list
        List of numbers with the number of neurons in the decoding part.
    optimizer: Union[str, Optimizer]
        If str then the name of the optimizer must be provided (e.x. "adam").
        The arguments of the optimizer can be supplied in optimization_kwargs.
        If a Keras optimizer call the instance of the respective
        class (e.x. Adam(lr=0.01,beta_1=0.9, beta_2=0.999)).  If no arguments are
        provided Keras default values will be set.
    optimizer_kwargs: Dict[str, Any]
        The arguments for the chosen optimizer. If not provided Keras'
        default values will be used.
    compile_kwargs: Dict[str, Any]
        Parameters to pass to ``keras.Model.compile``

    Returns
    -------
    keras.models.Sequential
        Returns Keras sequential model.

    """

    input_dim = n_features

    if len(encoding_dim) != len(encoding_func):
        raise ValueError(
            f"Number of layers ({len(encoding_dim)}) and number of functions ({len(encoding_func)}) "
            f"must be equal for the encoder."
        )

    if len(decoding_dim) != len(decoding_func):
        raise ValueError(
            f"Number of layers ({len(decoding_dim)}) and number of functions ({len(decoding_func)}) "
            f"must be equal for the decoder."
        )

    model = KerasSequential()
    # encoding layers
    for i, (n_neurons, activation) in enumerate(zip(encoding_dim, encoding_func)):
        if i == 0:
            model.add(
                LSTM(
                    n_neurons,
                    activation=activation,
                    input_shape=(lookback_window, input_dim),
                    return_sequences=True,
                )
            )

        else:
            model.add(LSTM(n_neurons, activation=activation, return_sequences=True))

    # decoding layers
    for i, (n_neurons, activation) in enumerate(zip(decoding_dim, decoding_func)):
        if i != len(decoding_dim) - 1:
            model.add(LSTM(n_neurons, activation=activation, return_sequences=True))
        else:
            model.add(LSTM(n_neurons, activation=activation, return_sequences=False))

    # output layer
    if isinstance(optimizer, str):
        Optim = getattr(keras.optimizers, optimizer)
        optimizer = Optim(**optimizer_kwargs)

    model.add(Dense(units=input_dim, activation=out_func))

    # Update kwargs and compile model
    compile_kwargs.setdefault("loss", "mse")
    compile_kwargs.update({"optimizer": optimizer})
    model.compile(**compile_kwargs)

    return model


@register_model_builder(type="KerasLSTMAutoEncoder")
@register_model_builder(type="KerasLSTMForecast")
def lstm_symmetric(
    n_features: int,
    lookback_window: int = 1,
    dims: Tuple[int, ...] = (256, 128, 64),
    funcs: Tuple[str, ...] = ("tanh", "tanh", "tanh"),
    out_func: str = "linear",
    optimizer: Union[str, Optimizer] = "adam",
    optimizer_kwargs: Dict[str, Any] = dict(),
    compile_kwargs: Dict[str, Any] = dict(),
    **kwargs,
) -> keras.models.Sequential:
    """
    Builds a symmetrical lstm model

    Parameters
    ----------
    n_features: int
         Number of input and output neurons
    lookback_window: int
        Number of timesteps used to train the model.
        One timestep = sample contains current observation.
        Two timesteps = sample contains current and previous observation.
        ...
    dims: Tuple[int,...]
         Number of neurons per layers for the encoder, reversed for the decoder.
         Must have len > 0
    funcs: List[str]
        Activation functions for the internal layers
    out_func: str
        Activation function for the output Dense layer.
    optimizer: Union[str, Optimizer]
        If str then the name of the optimizer must be provided (e.x. "adam").
        The arguments of the optimizer can be supplied in optimization_kwargs.
        If a Keras optimizer call the instance of the respective
        class (e.x. Adam(lr=0.01,beta_1=0.9, beta_2=0.999)).  If no arguments are
        provided Keras default values will be set.
    optimizer_kwargs: Dict[str, Any]
        The arguments for the chosen optimizer. If not provided Keras'
        default values will be used.
    compile_kwargs: Dict[str, Any]
        Parameters to pass to ``keras.Model.compile``

    Returns
    -------
    keras.models.Sequential
        Returns Keras sequential model.
    """

    if len(dims) == 0:
        raise ValueError("Parameter dims must have len > 0")
    return lstm_model(
        n_features,
        lookback_window,
        dims,
        dims[::-1],
        funcs,
        funcs[::-1],
        out_func,
        optimizer,
        optimizer_kwargs,
        compile_kwargs,
        **kwargs,
    )


@register_model_builder(type="KerasLSTMAutoEncoder")
@register_model_builder(type="KerasLSTMForecast")
def lstm_hourglass(
    n_features: int,
    lookback_window: int = 1,
    encoding_layers: int = 3,
    compression_factor: float = 0.5,
    func: str = "tanh",
    out_func: str = "linear",
    optimizer: Union[str, Optimizer] = "adam",
    optimizer_kwargs: Dict[str, Any] = dict(),
    compile_kwargs: Dict[str, Any] = dict(),
    **kwargs,
) -> keras.models.Sequential:

    """

    Builds an hourglass shaped neural network, with decreasing number of neurons
    as one gets deeper into the encoder network and increasing number
    of neurons as one gets out of the decoder network.


    Parameters
    ----------
    n_features: int
        Number of input and output neurons
    encoding_layers: int
        Number of layers from the input layer (exclusive) to the
        narrowest layer (inclusive). Must be > 0. The total nr of layers
        including input and output layer will be 2*encoding_layers + 1.
     compression_factor: float
        How small the smallest layer is as a ratio of n_features
        (smallest layer is rounded up to nearest integer). Must satisfy
        0 <= compression_factor <= 1.
    func: str
        Activation function for the internal layers
    out_func: str
        Activation function for the output Dense layer.
    optimizer: Union[str, Optimizer]
        If str then the name of the optimizer must be provided (e.x. "adam").
        The arguments of the optimizer can be supplied in optimization_kwargs.
        If a Keras optimizer call the instance of the respective
        class (e.x. Adam(lr=0.01,beta_1=0.9, beta_2=0.999)).  If no arguments are
        provided Keras default values will be set.
    optimizer_kwargs: Dict[str, Any]
        The arguments for the chosen optimizer. If not provided Keras'
        default values will be used.
    compile_kwargs: Dict[str, Any]
        Parameters to pass to ``keras.Model.compile``


    Returns
    -------
    keras.models.Sequential

    Examples
    --------
    >>> model = lstm_hourglass(10)
    >>> len(model.layers)
    7
    >>> [model.layers[i].units for i in range(len(model.layers))]
    [8, 7, 5, 5, 7, 8, 10]
    >>> model = lstm_hourglass(5)
    >>> [model.layers[i].units for i in range(len(model.layers))]
    [4, 4, 3, 3, 4, 4, 5]
    >>> model = lstm_hourglass(10, compression_factor=0.2)
    >>> [model.layers[i].units for i in range(len(model.layers))]
    [7, 5, 2, 2, 5, 7, 10]
    >>> model = lstm_hourglass(10, encoding_layers=1)
    >>> [model.layers[i].units for i in range(len(model.layers))]
    [5, 5, 10]
    """
    dims = hourglass_calc_dims(compression_factor, encoding_layers, n_features)

    return lstm_symmetric(
        n_features,
        lookback_window,
        dims,
        [func] * len(dims),
        out_func,
        optimizer,
        optimizer_kwargs,
        compile_kwargs,
        **kwargs,
    )
