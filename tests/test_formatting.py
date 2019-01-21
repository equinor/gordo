# -*- coding: utf-8 -*-
import os
import unittest
import black
from click.testing import CliRunner


class FormattingTestCase(unittest.TestCase):
    def test_project_formatting(self):
        """
        Test existing code would require no re-formatting
        """
        project_path = os.path.join(os.path.dirname(__file__), "..")
        gordo_components_path = os.path.join(project_path, "gordo_components")
        tests_path = os.path.join(project_path, "tests")
        runner = CliRunner()
        resp = runner.invoke(
            black.main,
            [
                "--check",
                "-v",
                gordo_components_path,
                tests_path,
                "--exclude",
                r".*_version.py",
            ],
        )
        self.assertEqual(
            resp.exit_code,
            0,
            msg=f"Black would still reformat one or ore files:\n{resp.output}",
        )
