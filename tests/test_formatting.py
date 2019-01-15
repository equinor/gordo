# -*- coding: utf-8 -*-

import unittest
import black
from click.testing import CliRunner


class FormattingTestCase(unittest.TestCase):
    def test_project_formatting(self):
        """
        Test existing code would require no re-formatting
        """
        runner = CliRunner()
        resp = runner.invoke(
            black.main,
            [
                "--check",
                "-v",
                "gordo_components",
                "tests",
                "--exclude",
                r".*_version.py",
            ],
        )
        self.assertEqual(
            resp.exit_code,
            0,
            msg=f"Black would still reformat one or ore files:\n{resp.output}",
        )
