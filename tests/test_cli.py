# -*- coding: utf-8 -*-

import os
import unittest
import logging
import tempfile

from click.testing import CliRunner

from gordo_components import cli
from tests.utils import temp_env_vars

import json


logger = logging.getLogger(__name__)


class CliTestCase(unittest.TestCase):
    """
    Test the expected usability of the CLI interface
    """

    def setUp(self):
        self.runner = CliRunner()

    def test_build_env_args(self):
        """
        Instead of passing OUTPUT_DIR directly to CLI, should be able to 
        read environment variables
        """

        model_config = {
            "gordo_components.model.models.KerasAutoEncoder": {
                "kind": "feedforward_symetric"
            }
        }

        logger.info(f"MODEL_CONFIG={json.dumps(model_config)}")

        with tempfile.TemporaryDirectory() as tmpdir:
            with temp_env_vars(
                OUTPUT_DIR=tmpdir,
                TRAIN_START_DATE="2015-01-01",
                TRAIN_END_DATE="2015-06-01",
                DATA_CONFIG='{"type": "RandomDataset"}',
                MODEL_CONFIG=json.dumps(model_config),
            ):
                result = self.runner.invoke(cli.gordo, ["build"])

            self.assertEqual(result.exit_code, 0, msg=f"Command failed: {result}")
            self.assertTrue(
                os.path.exists("/tmp/model-location.txt"),
                msg='Building was supposed to create a "model-location.txt", but it did not!',
            )
