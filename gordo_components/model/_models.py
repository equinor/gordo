# -*- coding: utf-8 -*-

import json
import logging
import os

import keras
import numpy as np
import pandas as pd
from keras import regularizers
from keras.layers import Input, Dense
from keras.models import Sequential as KerasSequential
from sklearn.model_selection import train_test_split

from gordo_components.model.base import GordoBaseModel


class GordoKerasModel(GordoBaseModel):
    """ 
    The Keras Model
    """
    _model = None
    _input_dim = None

    def __init__(self, n_features, **kwargs):
        """
        Builds a customized keras neural network auto-encoder based on a config dict
        Args:
            n_features: int - Number of features the dataset X will contain.
            kwargs: dict - With key indicating the following:
                * input_dim: input shape on the 1st axis, ie (50,)
                * enc_dim: List of numbers with the number of neurons in the encoding part
                * dec_dim: List of numbers with the number of neurons in the decoding part
                * enc_func: Activation functions for the encoder part
                * dec_func: Activation functions for the decoder part
        Returns:
            GordoKerasModel()
        """
        input_dim     = n_features
        encoding_dim  = kwargs.get('enc_dim', [256, 128, 64])
        decoding_dim  = kwargs.get('dec_dim', [64, 128, 256])
        encoding_func = kwargs.get('enc_func', ['relu', 'relu', 'relu'])
        decoding_func = kwargs.get('dec_func', ['relu', 'relu', 'tanh'])

        encoding_layers = len(encoding_dim)
        decoding_layers = len(decoding_dim)

        model = KerasSequential()

        if encoding_layers != len(encoding_func):
            raise ValueError(
                "Number of layers ({}) and number of functions ({}) must be equal for the encoder.".format(
                    encoding_layers, len(encoding_func)))

        if decoding_layers != len(decoding_func):
            raise ValueError(
                "Number of layers ({}) and number of functions ({}) must be equal for the decoder.".format(
                    decoding_layers, len(decoding_func)))

        # Add encoding layers
        for i in range(encoding_layers):
            if i == 0:
                model.add(
                    Dense(
                        input_dim=input_dim,
                        units=encoding_dim[i], 
                        activation=encoding_func[i]
                    )
                )
            else:
                model.add(
                    Dense(
                        units=encoding_dim[i], 
                        activation=encoding_func[i],
                        activity_regularizer=regularizers.l1(10e-5)
                    )
                )

        # Add decoding layers
        for i in range(decoding_layers):
            model.add( 
                Dense(
                    units=decoding_dim[i], 
                    activation=decoding_func[i]
                )
            )

        # Final output layer
        model.add( 
            Dense(input_dim, activation='tanh')
        )

        model.compile(optimizer='adam',
                      loss='mean_squared_error',
                      metrics=['accuracy'])

        
        self._input_dim = input_dim
        self._model = model

    @property
    def model(self):
        return self._model

    def fit(self, X, y):
        return NotImplemented

    def predict(self, X):
        return NotImplemented
