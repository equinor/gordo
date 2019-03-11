# -*- coding: utf-8 -*-

import unittest
from unittest import mock

from gordo_components.model.factories.lstm_autoencoder import (
    lstm_symmetric,
    lstm_hourglass,
)


def lstm_model_mocker(
    n_features: int,
    lookback_window,
    enc_dim,
    dec_dim,
    enc_func,
    dec_func,
    out_func,
    optimizer,
    optimizer_kwargs,
    loss,
):
    return (
        n_features,
        lookback_window,
        enc_dim,
        dec_dim,
        enc_func,
        dec_func,
        out_func,
        optimizer,
        optimizer_kwargs,
        loss,
    )


class LSTMAutoEncoderTestCase(unittest.TestCase):
    @mock.patch(
        "gordo_components.model.factories.lstm_autoencoder.lstm_model",
        side_effect=lstm_model_mocker,
    )
    def test_lstm_symmetric_basic(self, _):
        """
        Test that lstm_symmetric calls lstm_model correctly
        """
        (
            n_features,
            lookback_window,
            enc_dim,
            dec_dim,
            enc_func,
            dec_func,
            out_func,
            optimizer,
            optimizer_kwargs,
            loss,
        ) = lstm_symmetric(
            5,
            3,
            [4, 3, 2, 1],
            ["relu", "relu", "tanh", "tanh"],
            "relu",
            "sgd",
            {"lr": 0.01},
            "mse",
        )
        self.assertEqual(n_features, 5)
        self.assertEqual(enc_dim, [4, 3, 2, 1])
        self.assertEqual(dec_dim, [1, 2, 3, 4])
        self.assertEqual(enc_func, ["relu", "relu", "tanh", "tanh"])
        self.assertEqual(dec_func, ["tanh", "tanh", "relu", "relu"])
        self.assertEqual(out_func, "relu")
        self.assertEqual(optimizer, "sgd")
        self.assertEqual(optimizer_kwargs, {"lr": 0.01})
        self.assertEqual(loss, "mse")

    def test_lstm_symmetric_checks_dims(self):
        """
        Test that lstm_symmetric validates parameter requirements
        """
        with self.assertRaises(ValueError):
            lstm_symmetric(4, dims=[], funcs=[])

    @mock.patch(
        "gordo_components.model.factories.lstm_autoencoder.lstm_model",
        side_effect=lstm_model_mocker,
    )
    def test_lstm_hourglass_basic(self, _):
        """
        Test that lstm_hourglass calls lstm_model correctly
        """
        (
            n_features,
            lookback_window,
            enc_dim,
            dec_dim,
            enc_func,
            dec_func,
            out_func,
            optimizer,
            optimizer_kwargs,
            loss,
        ) = lstm_hourglass(10)
        self.assertEqual(n_features, 10)
        self.assertEqual(enc_dim, [8, 7, 5])
        self.assertEqual(dec_dim, [5, 7, 8])
        self.assertEqual(enc_func, ["relu", "relu", "relu"])
        self.assertEqual(dec_func, ["relu", "relu", "relu"])
        self.assertEqual(out_func, "linear")
        self.assertEqual(optimizer, "adam")
        self.assertEqual(optimizer_kwargs, dict())
        self.assertEqual(loss, "mse")

        (
            n_features,
            lookback_window,
            enc_dim,
            dec_dim,
            enc_func,
            dec_func,
            out_func,
            optimizer,
            optimizer_kwargs,
            loss,
        ) = lstm_hourglass(
            3,
            func="tanh",
            out_func="relu",
            optimizer="sgd",
            optimizer_kwargs={"lr": 0.03},
            loss="mae",
        )
        self.assertEqual(n_features, 3)
        self.assertEqual(enc_dim, [3, 2, 2])
        self.assertEqual(dec_dim, [2, 2, 3])
        self.assertEqual(enc_func, ["tanh", "tanh", "tanh"])
        self.assertEqual(dec_func, ["tanh", "tanh", "tanh"])
        self.assertEqual(out_func, "relu")
        self.assertEqual(optimizer, "sgd")
        self.assertEqual(optimizer_kwargs, {"lr": 0.03})
        self.assertEqual(loss, "mae")

        (
            n_features,
            lookback_window,
            enc_dim,
            dec_dim,
            enc_func,
            dec_func,
            out_func,
            optimizer,
            optimizer_kwargs,
            loss,
        ) = lstm_hourglass(10, compression_factor=0.3)
        self.assertEqual(n_features, 10)
        self.assertEqual(enc_dim, [8, 5, 3])
        self.assertEqual(dec_dim, [3, 5, 8])
        self.assertEqual(enc_func, ["relu", "relu", "relu"])
        self.assertEqual(dec_func, ["relu", "relu", "relu"])
        self.assertEqual(out_func, "linear")
        self.assertEqual(optimizer, "adam")
        self.assertEqual(optimizer_kwargs, dict())
        self.assertEqual(loss, "mse")

    @mock.patch(
        "gordo_components.model.factories.lstm_autoencoder.lstm_model",
        side_effect=lstm_model_mocker,
    )
    def test_lstm_hourglass_compression_factors(self, _):
        """
        Test that lstm_hourglass handles compression_factor=1 and
        compression_factor=0 correctly.
        """

        # compression_factor=1 is no compression
        (
            n_features,
            lookback_window,
            enc_dim,
            dec_dim,
            enc_func,
            dec_func,
            out_func,
            optimizer,
            optimizer_kwargs,
            loss,
        ) = lstm_hourglass(10, compression_factor=1)
        self.assertEqual(n_features, 10)
        self.assertEqual(enc_dim, [10, 10, 10])
        self.assertEqual(dec_dim, [10, 10, 10])
        self.assertEqual(enc_func, ["relu", "relu", "relu"])
        self.assertEqual(dec_func, ["relu", "relu", "relu"])
        self.assertEqual(out_func, "linear")
        self.assertEqual(optimizer, "adam")
        self.assertEqual(optimizer_kwargs, dict())
        self.assertEqual(loss, "mse")

        # compression_factor=0 has smallest layer with 1 node.
        (
            n_features,
            lookback_window,
            enc_dim,
            dec_dim,
            enc_func,
            dec_func,
            out_func,
            optimizer,
            optimizer_kwargs,
            loss,
        ) = lstm_hourglass(100000, compression_factor=0)
        self.assertEqual(n_features, 100000)
        self.assertEqual(enc_dim, [66667, 33334, 1])
        self.assertEqual(dec_dim, [1, 33334, 66667])
        self.assertEqual(enc_func, ["relu", "relu", "relu"])
        self.assertEqual(dec_func, ["relu", "relu", "relu"])
        self.assertEqual(out_func, "linear")
        self.assertEqual(optimizer, "adam")
        self.assertEqual(optimizer_kwargs, dict())
        self.assertEqual(loss, "mse")

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
