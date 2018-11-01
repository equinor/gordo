# -*- coding: utf-8 -*-

from typing import List

from keras import regularizers
from keras.layers import Input, Dense
from keras.models import Sequential as KerasSequential


class KerasFeedForwardFactory:
    """
    This class contains methods for producing special Keras models
    """

    @staticmethod
    def build_feedforward_symetric_model(
        n_features: int = None,
        enc_dim: List[int] = None,
        dec_dim: List[int] = None,
        enc_func: List[str] = None,
        dec_func: List[str] = None,
        **kwargs):
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
        encoding_dim  = enc_dim or [256, 128, 64]
        decoding_dim  = dec_dim or [64, 128, 256]
        encoding_func = enc_func or ['relu', 'relu', 'relu']
        decoding_func = dec_func or ['relu', 'relu', 'tanh']

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
        return model
