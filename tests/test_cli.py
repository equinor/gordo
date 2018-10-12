# -*- coding: utf-8 -*-

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
            MODEL_CONFIG='{"type": "keras", "epochs": 5}'):
            result = self.runner.invoke(cli.gordo, ['build'])
        self.assertEqual(result.exit_code, 0)        
