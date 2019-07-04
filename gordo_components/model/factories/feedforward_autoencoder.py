# -*- coding: utf-8 -*-

from typing import List, Tuple, Dict, Any

from keras import regularizers
from keras.layers import Dense
import keras

from keras.models import Sequential as KerasSequential
from gordo_components.model.register import register_model_builder
from gordo_components.model.factories.model_factories_utils import hourglass_calc_dims


@register_model_builder(type="KerasAutoEncoder")
def feedforward_model(
    n_features: int,
    enc_dim: List[int] = None,
    dec_dim: List[int] = None,
    enc_func: List[str] = None,
    dec_func: List[str] = None,
    out_func: str = "linear",
    compile_kwargs: Dict[str, Any] = dict(),
    **kwargs,
) -> keras.models.Sequential:
    """
    Builds a customized keras neural network auto-encoder based on a config dict

    Parameters
    ----------
    n_features: int
        Number of features the dataset X will contain.
    enc_dim: list
        List of numbers with the number of neurons in the encoding part
    dec_dim: list
        List of numbers with the number of neurons in the decoding part
    enc_func: list
        Activation functions for the encoder part
    dec_func: list
        Activation functions for the decoder part
    out_func: str
        Activation function for the output layer
    compile_kwargs: Dict[str, Any]
        Parameters to pass to ``keras.Model.compile``

    Returns
    -------
    keras.models.Sequential

    """
    input_dim = n_features
    encoding_dim = enc_dim or [256, 128, 64]
    decoding_dim = dec_dim or [64, 128, 256]
    encoding_func = enc_func or ["tanh", "tanh", "tanh"]
    decoding_func = dec_func or ["tanh", "tanh", "tanh"]

    model = KerasSequential()

    if len(encoding_dim) != len(encoding_func):
        raise ValueError(
            f"Number of layers ({len(encoding_dim)}) and number of functions ({len(encoding_func)}) must be equal for the encoder."
        )

    if len(decoding_dim) != len(decoding_func):
        raise ValueError(
            f"Number of layers ({len(decoding_dim)}) and number of functions ({len(decoding_func)}) must be equal for the decoder."
        )

    # Add encoding layers
    for i, (units, activation) in enumerate(zip(encoding_dim, encoding_func)):

        args = {"units": units, "activation": activation}

        if i == 0:
            args["input_dim"] = input_dim
        else:
            args["activity_regularizer"] = regularizers.l1(10e-5)

        model.add(Dense(**args))

    # Add decoding layers
    for i, (units, activation) in enumerate(zip(decoding_dim, decoding_func)):
        model.add(Dense(units=units, activation=activation))

    # Final output layer
    model.add(Dense(input_dim, activation=out_func))

    # Set some pre-determined default compile kwargs.
    compile_kwargs.setdefault("optimizer", "adam")
    compile_kwargs.setdefault("loss", "mean_squared_error")
    compile_kwargs.setdefault("metrics", ["accuracy"])

    model.compile(**compile_kwargs)
    return model


@register_model_builder(type="KerasAutoEncoder")
def feedforward_symmetric(
    n_features: int,
    dims: Tuple[int, ...] = (256, 128, 64),
    funcs: Tuple[str, ...] = ("tanh", "tanh", "tanh"),
    compile_kwargs: Dict[str, Any] = dict(),
    **kwargs,
) -> keras.models.Sequential:
    """
    Builds a symmetrical feedforward model

    Parameters
    ----------
    n_features: int
         Number of input and output neurons
    dim: List[int]
         Number of neurons per layers for the encoder, reversed for the decoder.
         Must have len > 0
    funcs: List[str]
        Activation functions for the internal layers
    compile_kwargs: Dict[str, Any]
        Parameters to pass to ``keras.Model.compile``

    Returns
    -------
    keras.models.Sequential

    """
    if len(dims) == 0:
        raise ValueError("Parameter dims must have len > 0")
    return feedforward_model(
        n_features,
        enc_dim=dims,
        dec_dim=dims[::-1],
        enc_func=funcs,
        dec_func=funcs[::-1],
        compile_kwargs=compile_kwargs,
        **kwargs,
    )


@register_model_builder(type="KerasAutoEncoder")
def feedforward_hourglass(
    n_features: int,
    encoding_layers: int = 3,
    compression_factor: float = 0.5,
    func: str = "tanh",
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
    compile_kwargs: Dict[str, Any]
        Parameters to pass to ``keras.Model.compile``

    Notes
    -----
    The resulting model will look like this when n_features = 10, encoding_layers= 3,
    and compression_factor = 0.3::

                * * * * * * * * * *
                  * * * * * * * *
                     * * * * *
                       * * *
                       * * *
                     * * * * *
                  * * * * * * * *
                * * * * * * * * * *


    Returns
    -------
    keras.models.Sequential

    Examples
    --------
    >>> model = feedforward_hourglass(10)
    >>> len(model.layers)
    7
    >>> [model.layers[i].units for i in range(len(model.layers))]
    [8, 7, 5, 5, 7, 8, 10]
    >>> model = feedforward_hourglass(5)
    >>> [model.layers[i].units for i in range(len(model.layers))]
    [4, 4, 3, 3, 4, 4, 5]
    >>> model = feedforward_hourglass(10, compression_factor=0.2)
    >>> [model.layers[i].units for i in range(len(model.layers))]
    [7, 5, 2, 2, 5, 7, 10]
    >>> model = feedforward_hourglass(10, encoding_layers=1)
    >>> [model.layers[i].units for i in range(len(model.layers))]
    [5, 5, 10]
    """
    dims = hourglass_calc_dims(compression_factor, encoding_layers, n_features)

    return feedforward_symmetric(
        n_features,
        dims=dims,
        funcs=[func] * len(dims),
        compile_kwargs=compile_kwargs,
        **kwargs,
    )
