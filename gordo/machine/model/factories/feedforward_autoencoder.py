# -*- coding: utf-8 -*-

from typing import Tuple, Dict, Any, Union

from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense
from tensorflow import keras

from tensorflow.keras.models import Sequential as KerasSequential
from gordo.machine.model.register import register_model_builder
from gordo.machine.model.factories.utils import hourglass_calc_dims, check_dim_func_len


@register_model_builder(type="KerasAutoEncoder")
def feedforward_model(
    n_features: int,
    n_features_out: int = None,
    encoding_dim: Tuple[int, ...] = (256, 128, 64),
    encoding_func: Tuple[str, ...] = ("tanh", "tanh", "tanh"),
    decoding_dim: Tuple[int, ...] = (64, 128, 256),
    decoding_func: Tuple[str, ...] = ("tanh", "tanh", "tanh"),
    out_func: str = "linear",
    optimizer: Union[str, Optimizer] = "Adam",
    optimizer_kwargs: Dict[str, Any] = dict(),
    compile_kwargs: Dict[str, Any] = dict(),
    **kwargs,
) -> keras.models.Sequential:
    """
    Builds a customized keras neural network auto-encoder based on a config dict

    Parameters
    ----------
    n_features: int
        Number of features the dataset X will contain.
    n_features_out: Optional[int]
        Number of features the model will output, default to ``n_features``.
    encoding_dim: tuple
        Tuple of numbers with the number of neurons in the encoding part.
    decoding_dim: tuple
        Tuple of numbers with the number of neurons in the decoding part.
    encoding_func: tuple
        Activation functions for the encoder part.
    decoding_func: tuple
        Activation functions for the decoder part.
    out_func: str
        Activation function for the output layer
    optimizer: Union[str, Optimizer]
        If str then the name of the optimizer must be provided (e.x. "Adam").
        The arguments of the optimizer can be supplied in optimize_kwargs.
        If a Keras optimizer call the instance of the respective
        class (e.x. Adam(lr=0.01,beta_1=0.9, beta_2=0.999)).  If no arguments are
        provided Keras default values will be set.
    optimizer_kwargs: Dict[str, Any]
        The arguments for the chosen optimizer. If not provided Keras'
        default values will be used.
    compile_kwargs: Dict[str, Any]
        Parameters to pass to ``keras.Model.compile``.

    Returns
    -------
    keras.models.Sequential

    """

    input_dim = n_features
    n_features_out = n_features_out or n_features

    check_dim_func_len("encoding", encoding_dim, encoding_func)
    check_dim_func_len("decoding", decoding_dim, decoding_func)

    model = KerasSequential()

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

    # Instantiate optimizer with kwargs
    if isinstance(optimizer, str):
        Optim = getattr(keras.optimizers, optimizer)
        optimizer = Optim(**optimizer_kwargs)

    # Final output layer
    model.add(Dense(n_features_out, activation=out_func))

    # Set some pre-determined default compile kwargs.
    compile_kwargs.update({"optimizer": optimizer})
    compile_kwargs.setdefault("loss", "mean_squared_error")
    compile_kwargs.setdefault("metrics", ["accuracy"])

    model.compile(**compile_kwargs)
    return model


@register_model_builder(type="KerasAutoEncoder")
def feedforward_symmetric(
    n_features: int,
    n_features_out: int = None,
    dims: Tuple[int, ...] = (256, 128, 64),
    funcs: Tuple[str, ...] = ("tanh", "tanh", "tanh"),
    optimizer: Union[str, Optimizer] = "Adam",
    optimizer_kwargs: Dict[str, Any] = dict(),
    compile_kwargs: Dict[str, Any] = dict(),
    **kwargs,
) -> keras.models.Sequential:
    """
    Builds a symmetrical feedforward model

    Parameters
    ----------
    n_features: int
         Number of input and output neurons.
    n_features_out: Optional[int]
        Number of features the model will output, default to ``n_features``.
    dim: List[int]
         Number of neurons per layers for the encoder, reversed for the decoder.
         Must have len > 0.
    funcs: List[str]
        Activation functions for the internal layers
    optimizer: Union[str, Optimizer]
        If str then the name of the optimizer must be provided (e.x. "Adam").
        The arguments of the optimizer can be supplied in optimization_kwargs.
        If a Keras optimizer call the instance of the respective
        class (e.x. Adam(lr=0.01,beta_1=0.9, beta_2=0.999)).  If no arguments are
        provided Keras default values will be set.
    optimizer_kwargs: Dict[str, Any]
        The arguments for the chosen optimizer. If not provided Keras'
        default values will be used.
    compile_kwargs: Dict[str, Any]
        Parameters to pass to ``keras.Model.compile``.

    Returns
    -------
    keras.models.Sequential

    """
    if len(dims) == 0:
        raise ValueError("Parameter dims must have len > 0")
    return feedforward_model(
        n_features,
        n_features_out,
        encoding_dim=dims,
        decoding_dim=dims[::-1],
        encoding_func=funcs,
        decoding_func=funcs[::-1],
        optimizer=optimizer,
        optimizer_kwargs=optimizer_kwargs,
        compile_kwargs=compile_kwargs,
        **kwargs,
    )


@register_model_builder(type="KerasAutoEncoder")
def feedforward_hourglass(
    n_features: int,
    n_features_out: int = None,
    encoding_layers: int = 3,
    compression_factor: float = 0.5,
    func: str = "tanh",
    optimizer: Union[str, Optimizer] = "Adam",
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
        Number of input and output neurons.
    n_features_out: Optional[int]
        Number of features the model will output, default to ``n_features``.
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
    optimizer: Union[str, Optimizer]
        If str then the name of the optimizer must be provided (e.x. "Adam").
        The arguments of the optimizer can be supplied in optimization_kwargs.
        If a Keras optimizer call the instance of the respective
        class (e.x. Adam(lr=0.01,beta_1=0.9, beta_2=0.999)).  If no arguments are
        provided Keras default values will be set.
    optimizer_kwargs: Dict[str, Any]
        The arguments for the chosen optimizer. If not provided Keras'
        default values will be used.
    compile_kwargs: Dict[str, Any]
        Parameters to pass to ``keras.Model.compile``.

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
        n_features_out,
        dims=dims,
        funcs=tuple([func] * len(dims)),
        optimizer=optimizer,
        optimizer_kwargs=optimizer_kwargs,
        compile_kwargs=compile_kwargs,
        **kwargs,
    )
