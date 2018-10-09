# -*- coding: utf-8 -*-

import unittest
import os

from click.testing import CliRunner

from gordo_flow import cli



class CliTestCase(unittest.TestCase):

    def setUp(self):
        self.runner = CliRunner()

    def test_build_no_args(self):
        """
        OUTPUT_DIR is a required arg to 'gordo-flow build'
        """
        result = self.runner.invoke(cli.gordo, ['build'])
        self.assertTrue(
            'Error: Missing argument "OUTPUT_DIR".' in result.output,
            msg='Unexpected output for "gordo-flow build": {}'.format(result.output)
        )

    def test_build_env_args(self):
        """
        Instead of passing OUTPUT_DIR directly to CLI, should be able to 
        read environment variables
        """
        with temp_env_vars(OUTPUT_DIR='/tmp'):
            result = self.runner.invoke(cli.gordo, ['build'])

        self.assertEqual(result.exit_code, 0)
        
