# -*- coding: utf-8 -*-

import unittest
from unittest import mock

from gordo.machine.model.factories.feedforward_autoencoder import (
    feedforward_symmetric,
    feedforward_hourglass,
)


def feedforward_model_mocker(
    n_features: int,
    n_features_out: int,
    encoding_dim,
    decoding_dim,
    encoding_func,
    decoding_func,
    optimizer,
    optimizer_kwargs,
    compile_kwargs,
):
    return (
        n_features,
        n_features_out,
        encoding_dim,
        decoding_dim,
        encoding_func,
        decoding_func,
        optimizer,
        optimizer_kwargs,
        compile_kwargs,
    )


class FeedForwardAutoEncoderTestCase(unittest.TestCase):
    @mock.patch(
        "gordo.machine.model.factories.feedforward_autoencoder.feedforward_model",
        side_effect=feedforward_model_mocker,
    )
    def test_feedforward_symmetric_basic(self, _):
        """
        Test that feedforward_symmetric calls feedforward_model correctly
        """
        (
            n_features,
            n_features_out,
            encoding_dim,
            decoding_dim,
            encoding_func,
            decoding_func,
            optimizer,
            optimizer_kwargs,
            compile_kwargs,
        ) = feedforward_symmetric(
            5, 5, (4, 3, 2, 1), ("relu", "relu", "tanh", "tanh"), {}
        )
        self.assertEqual(n_features, 5)
        self.assertEqual(compile_kwargs, {})
        self.assertEqual(encoding_dim, (4, 3, 2, 1))
        self.assertEqual(decoding_dim, (1, 2, 3, 4))
        self.assertEqual(encoding_func, ("relu", "relu", "tanh", "tanh"))
        self.assertEqual(decoding_func, ("tanh", "tanh", "relu", "relu"))

    def test_feedforward_symmetric_checks_dims(self):
        """
        Test that feedforward_symmetric validates parameter requirements
        """
        with self.assertRaises(ValueError):
            feedforward_symmetric(4, (), ())

    @mock.patch(
        "gordo.machine.model.factories.feedforward_autoencoder.feedforward_model",
        side_effect=feedforward_model_mocker,
    )
    def test_feedforward_hourglass_basic(self, _):
        """
        Test that feedforward_hourglass calls feedforward_model correctly
        """
        (
            n_features,
            n_features_out,
            encoding_dim,
            decoding_dim,
            encoding_func,
            decoding_func,
            optimizer,
            optimizer_kwargs,
            compile_kwargs,
        ) = feedforward_hourglass(10, 10, func="relu")
        self.assertEqual(n_features, 10)
        self.assertEqual(n_features_out, 10)
        self.assertEqual(encoding_dim, (8, 7, 5))
        self.assertEqual(decoding_dim, (5, 7, 8))
        self.assertEqual(encoding_func, ("relu", "relu", "relu"))
        self.assertEqual(decoding_func, ("relu", "relu", "relu"))

        (
            n_features,
            n_features_out,
            encoding_dim,
            decoding_dim,
            encoding_func,
            decoding_func,
            optimizer,
            optimizer_kwargs,
            compile_kwargs,
        ) = feedforward_hourglass(3, 3)
        self.assertEqual(n_features, 3)
        self.assertEqual(n_features_out, 3)
        self.assertEqual(encoding_dim, (3, 2, 2))
        self.assertEqual(decoding_dim, (2, 2, 3))
        self.assertEqual(encoding_func, ("tanh", "tanh", "tanh"))
        self.assertEqual(decoding_func, ("tanh", "tanh", "tanh"))

        (
            n_features,
            n_features_out,
            encoding_dim,
            decoding_dim,
            encoding_func,
            decoding_func,
            optimizer,
            optimizer_kwargs,
            compile_kwargs,
        ) = feedforward_hourglass(10, 10, compression_factor=0.3)
        self.assertEqual(n_features, 10)
        self.assertEqual(n_features_out, 10)
        self.assertEqual(encoding_dim, (8, 5, 3))
        self.assertEqual(decoding_dim, (3, 5, 8))
        self.assertEqual(encoding_func, ("tanh", "tanh", "tanh"))
        self.assertEqual(decoding_func, ("tanh", "tanh", "tanh"))

    @mock.patch(
        "gordo.machine.model.factories.feedforward_autoencoder.feedforward_model",
        side_effect=feedforward_model_mocker,
    )
    def test_feedforward_hourglass_compression_factors(self, _):
        """
        Test that feedforward_hourglass handles compression_factor=1 and
        compression_factor=0 correctly.
        """

        # compression_factor=1 is no compression
        (
            n_features,
            n_features_out,
            encoding_dim,
            decoding_dim,
            encoding_func,
            decoding_func,
            optimizer,
            optimizer_kwargs,
            compile_kwargs,
        ) = feedforward_hourglass(10, 10, compression_factor=1)
        self.assertEqual(n_features, 10)
        self.assertEqual(n_features_out, 10)
        self.assertEqual(encoding_dim, (10, 10, 10))
        self.assertEqual(decoding_dim, (10, 10, 10))
        self.assertEqual(encoding_func, ("tanh", "tanh", "tanh"))
        self.assertEqual(decoding_func, ("tanh", "tanh", "tanh"))

        # compression_factor=0 has smallest layer with 1 node.
        (
            n_features,
            n_features_out,
            encoding_dim,
            decoding_dim,
            encoding_func,
            decoding_func,
            optimizer,
            optimizer_kwargs,
            compile_kwargs,
        ) = feedforward_hourglass(100000, 100000, compression_factor=0)
        self.assertEqual(n_features, 100000)
        self.assertEqual(n_features_out, 100000)
        self.assertEqual(encoding_dim, (66667, 33334, 1))
        self.assertEqual(decoding_dim, (1, 33334, 66667))
        self.assertEqual(encoding_func, ("tanh", "tanh", "tanh"))
        self.assertEqual(decoding_func, ("tanh", "tanh", "tanh"))

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
