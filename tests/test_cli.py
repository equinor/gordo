# -*- coding: utf-8 -*-

import os
import unittest

from click.testing import CliRunner

from gordo_components import cli
from tests.utils import temp_env_vars


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
        with temp_env_vars(
                OUTPUT_DIR='/tmp',
                DATA_CONFIG='{"type": "random"}',
                MODEL_CONFIG='{"type": "KerasModel", "kind": "feedforward_symetric", "epochs": 5}'):
            result = self.runner.invoke(cli.gordo, ['build'])

        self.assertEqual(result.exit_code, 0, msg=f"Command failed: {result.output}")
        self.assertTrue(os.path.exists('/tmp/model-location.txt'),
                        msg='Building was supposed to create a "model-location.txt", but it did not!')
