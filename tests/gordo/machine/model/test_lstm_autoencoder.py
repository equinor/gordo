# -*- coding: utf-8 -*-

import unittest

import pytest
from tensorflow.keras import optimizers

from gordo.machine.model.factories.lstm_autoencoder import (
    lstm_model,
    lstm_symmetric,
    lstm_hourglass,
)


class LSTMAutoEncoderTestCase(unittest.TestCase):
    def test_lstm_defaults(self):
        """
        Test that models with all defaults are created without failure and are equal
        """
        # Create models with default parameters
        base = lstm_model(4)
        symmetric = lstm_symmetric(4)
        hourglass = lstm_hourglass(4)

        # Ensure LSTM model layers are equal to base model layers
        for i in range(len(base.layers)):
            # Rename layers so as to not fail on names, only configuration
            config = base.layers[i].get_config().update({"name": "test"})
            symmetric_config = symmetric.layers[i].get_config().update({"name": "test"})
            hourglass_config = hourglass.layers[i].get_config().update({"name": "test"})
            assert config == symmetric_config
            assert config == hourglass_config

    def test_lstm_symmetric_checks_dims(self):
        """
        Test that lstm_symmetric validates parameter requirements
        """

        # Raise an error if empty dimension and function layers defined.
        with self.assertRaises(ValueError):
            lstm_symmetric(4, dims=(), funcs=())

        # Ensure failure with default 3 dimension layers when 2 function layers passed.
        with self.assertRaises(ValueError):
            lstm_symmetric(4, funcs=("tanh", "tanh"))

    def test_lstm_hourglass_basic(self):
        """
        Test that lstm_hourglass implements the correct parameters
        """

        model = lstm_hourglass(
            n_features=3,
            func="tanh",
            out_func="relu",
            optimizer="SGD",
            optimizer_kwargs={"lr": 0.02, "momentum": 0.001},
            compile_kwargs={"loss": "mae"},
        )

        # Ensure that the input dimension to Keras model matches the number of features.
        self.assertEqual(model.layers[0].input_shape[2], 3)

        # Ensure that the dimension of each encoding layer matches the expected dimension.
        self.assertEqual(
            [model.layers[i].input_shape[2] for i in range(1, 4)], [3, 2, 2]
        )

        # Ensure that the dimension of each decoding layer (excluding last decoding layer)
        # matches the expected dimension.
        self.assertEqual([model.layers[i].input_shape[2] for i in range(4, 6)], [2, 2])

        # Ensure that the dimension of last decoding layer matches the expected dimension.
        self.assertEqual(model.layers[6].input_shape[1], 3)

        # Ensure activation functions in the encoding part (layers 0-2)
        # match expected activation functions
        self.assertEqual(
            [model.layers[i].activation.__name__ for i in range(0, 3)],
            ["tanh", "tanh", "tanh"],
        )

        # Ensure activation functions in the decoding part (layers 3-5)
        # match expected activation functions
        self.assertEqual(
            [model.layers[i].activation.__name__ for i in range(3, 6)],
            ["tanh", "tanh", "tanh"],
        )

        # Ensure activation function for the output layer matches expected activation function
        self.assertEqual(model.layers[6].activation.__name__, "relu")

        # Assert that the expected Keras optimizer is used
        self.assertEqual(model.optimizer.__class__, optimizers.SGD)

        # Assert that the correct loss function is used.
        self.assertEqual(model.loss, "mae")

    def test_lstm_hourglass_checks_enc_layers(self):
        """
        Test that lstm_hourglass validates parameter requirements
        """
        with self.assertRaises(ValueError):
            lstm_hourglass(3, encoding_layers=0)

    def test_lstm_hourglass_checks_compression_factor(self):
        """
        Test that lstm_hourglass validates parameter requirements
        """
        with self.assertRaises(ValueError):
            lstm_hourglass(3, compression_factor=2)
        with self.assertRaises(ValueError):
            lstm_hourglass(3, compression_factor=-1)


@pytest.mark.parametrize("n_features", (5,))
@pytest.mark.parametrize("n_features_out", (10, 5, 2, 1))
def test_lstm_symmetric_basic(n_features, n_features_out):
    """
    Tests that lstm_symmetric implements the correct parameters
    """
    model = lstm_symmetric(
        n_features=n_features,
        n_features_out=n_features_out,
        lookback_window=3,
        dims=(4, 3, 2, 1),
        funcs=("relu", "relu", "tanh", "tanh"),
        out_func="linear",
        optimizer="SGD",
        optimizer_kwargs={"lr": 0.01},
        loss="mse",
    )

    # Ensure that the input dimension to Keras model matches the number of features.
    assert model.layers[0].input_shape[2] == n_features

    # Ensure that the dimension of each encoding layer matches the expected dimension.
    assert [model.layers[i].input_shape[2] for i in range(1, 5)] == [4, 3, 2, 1]

    # Ensure that the dimension of each decoding layer (excluding last decoding layer)
    # matches the expected dimension.
    assert [model.layers[i].input_shape[2] for i in range(5, 8)] == [1, 2, 3]

    # Ensure that the dimension of last decoding layer matches the expected dimension.
    assert model.layers[8].input_shape[1] == 4

    # Ensure activation functions in the encoding part (layers 0-3)
    # match expected activation functions.
    assert [model.layers[i].activation.__name__ for i in range(0, 4)] == [
        "relu",
        "relu",
        "tanh",
        "tanh",
    ]

    # Ensure activation functions in the decoding part (layers 4-7)
    # match expected activation functions.
    assert [model.layers[i].activation.__name__ for i in range(4, 8)] == [
        "tanh",
        "tanh",
        "relu",
        "relu",
    ]

    # Ensure activation function for the output layer matches expected activation function.
    assert model.layers[8].activation.__name__ == "linear"

    # Assert that the expected Keras optimizer is used
    assert model.optimizer.__class__ == optimizers.SGD

    # Assert that the correct loss function is used.
    assert model.loss == "mse"
