# -*- coding: utf-8 -*-

from typing import List

from keras import regularizers
from keras.layers import Dense
from keras.models import Sequential as KerasSequential

from gordo_components.model.register import register_model_builder


@register_model_builder(type="KerasAutoEncoder")
@register_model_builder(type="KerasModel")
def feedforward_symetric(
        n_features: int,
        enc_dim: List[int] = None,
        dec_dim: List[int] = None,
        enc_func: List[str] = None,
        dec_func: List[str] = None,
        **kwargs
):
    """
    Builds a customized keras neural network auto-encoder based on a config dict
    Args:
        n_features: int - Number of features the dataset X will contain.
        enc_dim: list -  List of numbers with the number of neurons in the encoding part
        dec_dim: list -  List of numbers with the number of neurons in the decoding part
        enc_func: list - Activation functions for the encoder part
        dec_func: list - Activation functions for the decoder part
    Returns:
        GordoKerasModel()
    """
    input_dim = n_features
    encoding_dim = enc_dim or [256, 128, 64]
    decoding_dim = dec_dim or [64, 128, 256]
    encoding_func = enc_func or ["relu", "relu", "relu"]
    decoding_func = dec_func or ["relu", "relu", "relu"]

    model = KerasSequential()

    if len(encoding_dim) != len(encoding_func):
        raise ValueError("Number of layers ({}) and number of functions ({}) must be equal for the encoder."
                         .format(len(encoding_dim), len(encoding_func))
                         )

    if len(decoding_dim) != len(decoding_func):
        raise ValueError("Number of layers ({}) and number of functions ({}) must be equal for the decoder."
                         .format(len(decoding_dim), len(decoding_func))
                         )

    # Add encoding layers
    for i, (units, activation) in enumerate(zip(encoding_dim, encoding_func)):

        args = {'units': units, 'activation': activation}

        if i == 0:
            args['input_dim'] = input_dim
        else:
            args['activity_regularizer'] = regularizers.l1(10e-5)

        model.add(Dense(**args))

    # Add decoding layers
    for i, (units, activation) in enumerate(zip(decoding_dim, decoding_func)):
        model.add(
            Dense(units=units, activation=activation)
        )

    # Final output layer
    model.add(Dense(input_dim, activation="tanh"))

    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["accuracy"])
    return model
