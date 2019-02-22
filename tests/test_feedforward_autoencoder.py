# -*- coding: utf-8 -*-

import unittest
from unittest import mock

from gordo_components.model.factories.feedforward_autoencoder import (
    feedforward_symmetric,
)


def feedforward_model_mocker(n_features: int, enc_dim, dec_dim, enc_func, dec_func):
    return n_features, enc_dim, dec_dim, enc_func, dec_func


class FeedForwardAutoEncoderTestCase(unittest.TestCase):
    @mock.patch(
        "gordo_components.model.factories.feedforward_autoencoder.feedforward_model",
        side_effect=feedforward_model_mocker,
    )
    def test_feedforward_symmetric_basic(self, _):
        """
        Test that feedforward_symmetric calls feedforward_model correctly
        """
        n_features, enc_dim, dec_dim, enc_func, dec_func = feedforward_symmetric(
            5, [4, 3, 2, 1], ["relu", "relu", "tanh", "tanh"]
        )
        self.assertEqual(n_features, 5)
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
