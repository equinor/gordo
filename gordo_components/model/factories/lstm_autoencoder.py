# -*- coding: utf-8 -*-

from typing import List, Tuple


from keras import regularizers
from keras.layers import Input, Dense, LSTM, TimeDistributed
from keras.models import Model

from gordo_components.model.register import register_model_builder


@register_model_builder(type="KerasAutoEncoder")
@register_model_builder(type="KerasModel")
def create_lstm_model(
        n_features: int,
        batch_size: int,
        encoder_units: List[int],
        decoder_units: List[int] = None,
        lstm_regularizer: regularizers.Regularizer = None,
        encoder_activation_functions: List[str] = None,
        encoder_recurrent_activation_functions: List[str] = None,
        encoder_dropout: List[float] = None,
        encoder_recurrent_dropout: List[float] = None,
        decoder_activation_functions: List[str] = None,
        decoder_recurrent_activation_functions: List[str] = None,
        decoder_dropout: List[float] = None,
        decoder_recurrent_dropout: List[float] = None,
        dense_activation: str = None,
        optimizer_method: str = None,
        loss_function: str = None,
        metric_function: List[str] = None,
        **kwargs
):

    """
    Builds a customized neural network auto-encoder
    Args:
        input_enc/dec: Number of input features for the encoder/decoder
        batch_size_enc/dec: Number of samples for the encoder/decoder
        lstm_regularizer: Regularization function used in the LSTM layers

        encoder_units: Number of unit per LSTM layer in the encoder
        encoder_activation_functions: Activation function of each LSTM layer in the encoder
        encoder_recurrent_activation_functions: Activation function for the recurrent gate of each LSTM layer in the encoder
        encoder_dropout: Dropout rate in the encoder layers
        encoder_recurrent_dropout: Dropout rate in the encoder layers for the recurrent gate

        decoder_units: Number of unit per LSTM layer in the decoder
        decoder_activation_functions: Activation function of each LSTM layer in the decoder
        decoder_recurrent_activation_functions: Activation function for the recurrent gate of each LSTM layer in the decoder
        decoder_dropout: Dropout rate in the decoder layers
        decoder_recurrent_dropout: Dropout rate in the decoder layers for the recurrent gate
        dense_activation: Activation function of the dense layer

        optimizer_method: Optimizer to compile the model with
        loss_function: Function to measure mis-match between simulated and target data
        metric_function: Assessment of model performance
    Returns:
        GordoKerasModel()
    """

    enc_units = encoder_units
    dec_units = decoder_units if decoder_units else encoder_units[::-1]

    enc_layers = len(enc_units)
    dec_layers = len(dec_units)

    reg = lstm_regularizer if lstm_regularizer else regularizers.l1(10e-5)

    # Encoder specifications
    enc_act = encoder_activation_functions \
        if encoder_activation_functions \
        else ['tanh']*enc_layers

    enc_rec_act = encoder_recurrent_activation_functions \
        if encoder_recurrent_activation_functions \
        else ['hard_sigmoid']*enc_layers

    enc_do = encoder_dropout \
        if encoder_dropout \
        else [0.2]*enc_layers

    enc_rec_do = encoder_recurrent_dropout \
        if encoder_recurrent_dropout \
        else [0.2]*enc_layers

    # Decoder specifications
    dec_act = decoder_activation_functions \
        if decoder_activation_functions \
        else ['tanh']*dec_layers

    dec_rec_act = decoder_recurrent_activation_functions \
        if decoder_recurrent_activation_functions \
        else ['hard_sigmoid']*dec_layers

    dec_do = decoder_dropout \
        if decoder_dropout \
        else [0.2]*dec_layers

    dec_rec_do = decoder_recurrent_dropout \
        if decoder_recurrent_dropout \
        else [0.2]*dec_layers

    # Dense layer specifications
    dense_act = dense_activation \
        if dense_activation \
        else 'linear'

    # Model training specifications
    opt = optimizer_method \
        if optimizer_method \
        else 'adam'

    loss = loss_function \
        if loss_function \
        else 'mean_squared_error'

    metric = metric_function \
        if metric_function \
        else ['accuracy']

    # Sanity check
    if not all(len(spec) == enc_layers for spec in [len(enc_units), len(enc_act), len(enc_rec_act), len(enc_do), len(enc_rec_do)]):
        raise ValueError(
            "Inconsistencies in lengths of encoder specifications.")

    if not all(len(spec) == dec_layers for spec in [len(dec_units), len(dec_act), len(dec_rec_act), len(dec_do), len(dec_rec_do)]):
        raise ValueError(
            "Inconsistencies in lengths of decoder specifications.")

    if enc_units[-1] != dec_units[0]:
        raise ValueError(
            "Encoder bottom layer must contain as many units as decoder top layer, for state initialization.")

    # Add encoding layers
    name = 'encoder_input_layer'
    input_enc_layer = Input(name=name, shape=(None, n_features))

    for i, (units, activation, recurrent_activation, dropout, recurrent_dropout) \
            in enumerate(zip(enc_units, enc_act, enc_rec_act, enc_do, enc_rec_do)):

        args = {'units': units,
                'name': 'encoder_layer_' + str(i + 1).zfill(2),
                'stateful': batch_size != 1,
                'batch_input_shape': (batch_size, None, n_features),
                'return_sequence': True,
                'return_state': True,
                'activation': activation,
                'recurrent_activation': recurrent_activation,
                'dropout': dropout,
                'recurrent_dropout': recurrent_dropout,
                'activity_regularizer': reg}

        encoder, state_h, state_c = LSTM(**args)(encoder)

    encoder_outputs = encoder
    encoder_states = [state_h, state_c]

    # Add decoding layers
    for i, (units, activation, recurrent_activation, dropout, recurrent_dropout) \
            in enumerate(zip(dec_units, dec_act, dec_rec_act, dec_do, dec_rec_do)):

        args = {'units': units,
                'name': 'decoder_layer_' + str(i + 1).zfill(2),
                'stateful': batch_size != 1,
                'batch_input_shape': (batch_size, None, n_features),
                'return_sequence': True,
                'return_state': True,
                'activation': activation,
                'recurrent_activation': recurrent_activation,
                'dropout': dropout,
                'recurrent_dropout': recurrent_dropout,
                'activity_regularizer': reg}

        input_args = {'inputs': decoder}

        if i == 0:
            input_args = {'inputs': encoder_outputs,
                          'initial_states': encoder_states}

        decoder = LSTM(**args)(**input_args)

    #Add dense layer
    name = 'dense_layer'
    output_layer = TimeDistributed(Dense(n_features, activation=dense_act), name=name)(decoder)

    model = Model(inputs=input_enc_layer, outputs=output_layer)
    model.compile(optimizer=opt,
                  loss=loss,
                  metrics=metric)

    return model
