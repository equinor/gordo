# -*- coding: utf-8 -*-

import unittest
from unittest import mock

from gordo_components.model.factories.feedforward_autoencoder import (
    feedforward_symmetric,
    feedforward_hourglass,
)


def feedforward_model_mocker(
    n_features: int, enc_dim, dec_dim, enc_func, dec_func, compile_kwargs
):
    return n_features, enc_dim, dec_dim, enc_func, dec_func, compile_kwargs


class FeedForwardAutoEncoderTestCase(unittest.TestCase):
    @mock.patch(
        "gordo_components.model.factories.feedforward_autoencoder.feedforward_model",
        side_effect=feedforward_model_mocker,
    )
    def test_feedforward_symmetric_basic(self, _):
        """
        Test that feedforward_symmetric calls feedforward_model correctly
        """
        n_features, enc_dim, dec_dim, enc_func, dec_func, compile_kwargs = feedforward_symmetric(
            5, [4, 3, 2, 1], ["relu", "relu", "tanh", "tanh"], {}
        )
        self.assertEqual(n_features, 5)
        self.assertEqual(compile_kwargs, {})
        self.assertEqual(enc_dim, [4, 3, 2, 1])
        self.assertEqual(dec_dim, [1, 2, 3, 4])
        self.assertEqual(enc_func, ["relu", "relu", "tanh", "tanh"])
        self.assertEqual(dec_func, ["tanh", "tanh", "relu", "relu"])

    def test_feedforward_symmetric_checks_dims(self):
        """
        Test that feedforward_symmetric validates parameter requirements
        """
        with self.assertRaises(ValueError):
            feedforward_symmetric(4, [], [])

    @mock.patch(
        "gordo_components.model.factories.feedforward_autoencoder.feedforward_model",
        side_effect=feedforward_model_mocker,
    )
    def test_feedforward_hourglass_basic(self, _):
        """
        Test that feedforward_hourglass calls feedforward_model correctly
        """
        n_features, enc_dim, dec_dim, enc_func, dec_func, compile_kwargs = feedforward_hourglass(
            10, func="relu"
        )
        self.assertEqual(n_features, 10)
        self.assertEqual(enc_dim, [8, 7, 5])
        self.assertEqual(dec_dim, [5, 7, 8])
        self.assertEqual(enc_func, ["relu", "relu", "relu"])
        self.assertEqual(dec_func, ["relu", "relu", "relu"])

        n_features, enc_dim, dec_dim, enc_func, dec_func, compile_kwargs = feedforward_hourglass(
            3
        )
        self.assertEqual(n_features, 3)
        self.assertEqual(enc_dim, [3, 2, 2])
        self.assertEqual(dec_dim, [2, 2, 3])
        self.assertEqual(enc_func, ["tanh", "tanh", "tanh"])
        self.assertEqual(dec_func, ["tanh", "tanh", "tanh"])

        n_features, enc_dim, dec_dim, enc_func, dec_func, compile_kwargs = feedforward_hourglass(
            10, compression_factor=0.3
        )
        self.assertEqual(n_features, 10)
        self.assertEqual(enc_dim, [8, 5, 3])
        self.assertEqual(dec_dim, [3, 5, 8])
        self.assertEqual(enc_func, ["tanh", "tanh", "tanh"])
        self.assertEqual(dec_func, ["tanh", "tanh", "tanh"])

    @mock.patch(
        "gordo_components.model.factories.feedforward_autoencoder.feedforward_model",
        side_effect=feedforward_model_mocker,
    )
    def test_feedforward_hourglass_compression_factors(self, _):
        """
        Test that feedforward_hourglass handles compression_factor=1 and
        compression_factor=0 correctly.
        """

        # compression_factor=1 is no compression
        n_features, enc_dim, dec_dim, enc_func, dec_func, compile_kwargs = feedforward_hourglass(
            10, compression_factor=1
        )
        self.assertEqual(n_features, 10)
        self.assertEqual(enc_dim, [10, 10, 10])
        self.assertEqual(dec_dim, [10, 10, 10])
        self.assertEqual(enc_func, ["tanh", "tanh", "tanh"])
        self.assertEqual(dec_func, ["tanh", "tanh", "tanh"])

        # compression_factor=0 has smallest layer with 1 node.
        n_features, enc_dim, dec_dim, enc_func, dec_func, compile_kwargs = feedforward_hourglass(
            100000, compression_factor=0
        )
        self.assertEqual(n_features, 100000)
        self.assertEqual(enc_dim, [66667, 33334, 1])
        self.assertEqual(dec_dim, [1, 33334, 66667])
        self.assertEqual(enc_func, ["tanh", "tanh", "tanh"])
        self.assertEqual(dec_func, ["tanh", "tanh", "tanh"])

    def test_feedforward_hourglass_checks_enc_layers(self):
        """
        Test that feedforward_hourglass validates parameter requirements
        """
        with self.assertRaises(ValueError):
            feedforward_hourglass(3, encoding_layers=0)

    def test_feedforward_hourglass_checks_compression_factor(self):
        """
        Test that feedforward_hourglass validates parameter requirements
        """
        with self.assertRaises(ValueError):
            feedforward_hourglass(3, compression_factor=2)
        with self.assertRaises(ValueError):
            feedforward_hourglass(3, compression_factor=-1)
