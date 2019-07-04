# -*- coding: utf-8 -*-

import unittest

import keras.backend as K
from keras import optimizers

from gordo_components.model.factories.lstm_autoencoder import (
    lstm_symmetric,
    lstm_hourglass,
)


class LSTMAutoEncoderTestCase(unittest.TestCase):
    def test_lstm_symmetric_basic(self):
        """
        Tests that lstm_symmetric implements the correct parameters
        """
        model = lstm_symmetric(
            n_features=5,
            lookback_window=3,
            dims=(4, 3, 2, 1),
            funcs=("relu", "relu", "tanh", "tanh"),
            out_func="linear",
            optimizer="sgd",
            optimizer_kwargs={"lr": 0.01},
            loss="mse",
        )

        # Ensure that the input dimension to Keras model matches the number of features.
        self.assertEqual(model.layers[0].input_shape[2], 5)

        # Ensure that the dimension of each encoding layer matches the expected dimension.
        self.assertEqual(
            [model.layers[i].input_shape[2] for i in range(1, 5)], [4, 3, 2, 1]
        )

        # Ensure that the dimension of each decoding layer (excluding last decoding layer)
        # matches the expected dimension.
        self.assertEqual(
            [model.layers[i].input_shape[2] for i in range(5, 8)], [1, 2, 3]
        )

        # Ensure that the dimension of last decoding layer matches the expected dimension.
        self.assertEqual(model.layers[8].input_shape[1], 4)

        # Ensure activation functions in the encoding part (layers 0-3)
        # match expected activation functions.
        self.assertEqual(
            [model.layers[i].activation.__name__ for i in range(0, 4)],
            ["relu", "relu", "tanh", "tanh"],
        )

        # Ensure activation functions in the decoding part (layers 4-7)
        # match expected activation functions.
        self.assertEqual(
            [model.layers[i].activation.__name__ for i in range(4, 8)],
            ["tanh", "tanh", "relu", "relu"],
        )

        # Ensure activation function for the output layer matches expected activation function.
        self.assertEqual(model.layers[8].activation.__name__, "linear")

        # Assert that the expected Keras optimizer is used
        self.assertEqual(model.optimizer.__class__, optimizers.SGD)

        # Assert equality of difference up to 7 decimal places
        # Note that AlmostEquality is used as Keras can use a value approximately equal
        # to the given learning rate rather than the exact value.
        self.assertAlmostEqual(K.eval(model.optimizer.lr), 0.01)

        # Assert that the correct loss function is used.
        self.assertEqual(model.loss, "mse")

    def test_lstm_symmetric_checks_dims(self):
        """
        Test that lstm_symmetric validates parameter requirements
        """
        with self.assertRaises(ValueError):
            lstm_symmetric(4, dims=[], funcs=[])

    def test_lstm_hourglass_basic(self):
        """
        Test that lstm_hourglass implements the correct parameters
        """

        model = lstm_hourglass(
            3,
            func="tanh",
            out_func="relu",
            optimizer="sgd",
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

        # Assert equality of difference up to 7 decimal places
        # Note that AlmostEquality is used as Keras can use a value approximately equal
        # to the given parameter rather than the exact value.
        self.assertAlmostEqual(K.eval(model.optimizer.lr), 0.02)
        self.assertAlmostEqual(K.eval(model.optimizer.momentum), 0.001)

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
