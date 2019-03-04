# -*- coding: utf-8 -*-

from typing import Tuple, Union, Dict, Any

import keras.optimizers
from keras.optimizers import Optimizer
from keras.layers import Dense, LSTM
from keras.models import Sequential as KerasSequential


from gordo_components.model.register import register_model_builder


@register_model_builder(type="KerasLSTMAutoEncoder")
def lstm_autoencoder(
    n_features: int,
    lookback_window: int = 1,
    encoding_dim: Tuple[int, ...] = (256, 128, 64),
    decoding_dim: Tuple[int, ...] = (64, 128, 256),
    encoding_func: Tuple[str, ...] = ("relu", "relu", "relu"),
    decoding_func: Tuple[str, ...] = ("relu", "relu", "relu"),
    out_func: str = "linear",
    optimizer: Union[str, Optimizer] = "adam",
    optimizer_kwargs: Dict[str, Any] = dict(),
    loss: str = "mse",
    **kwargs,
) -> keras.models.Sequential:
    """
    Builds a customized Keras LSTM neural network auto-encoder based on a config dict.

    Parameters
    ----------
    n_features: int
        Number of features the dataset X will contain.
    lookback_window: int
        Number of timestamps (lags) used to train the model.
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
    optimizer: str or keras optimizer
        If str then the name of the optimizer must be provided (e.x. "adam").
        The arguments of the optimizer can be supplied in optimization_kwargs.
        If a Keras optimizer call the instance of the respective
        class (e.x. Adam(lr=0.01,beta_1=0.9, beta_2=0.999)).  If no arguments are
        provided Keras default values will be set.
    optimizer_kwargs: dict
        The arguments for the chosen optimizer. If not provided Keras'
        default values will be used.
    loss: str
        Keras' supported loss functions (e.x. "mse", "MSE", "mean_squared_error"
                                              for mean squared error,
                                              "mae", "MAE", "mean_absolute_error"
                                              for mean absolute error,
                                              for other supported loss functions
                                              refer to https://keras.io/losses/).

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
    model.compile(optimizer=optimizer, loss=loss)
    return model
