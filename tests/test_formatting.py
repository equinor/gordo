# -*- coding: utf-8 -*-
import os
import sys
import unittest


class FormattingTestCase(unittest.TestCase):
    def test_formatting_black(self):
        project_path = os.path.join(os.path.dirname(__file__), "..")
        gordo_components_path = os.path.join(project_path, "gordo_components")
        tests_path = os.path.join(project_path, "tests")
        cmd = [
            sys.executable,
            "-m",
            "black",
            "--check",
            "-v",
            gordo_components_path,
            tests_path,
            "--exclude",
            r".*_version.py",
        ]
        exit_code = os.system(" ".join(cmd))
        self.assertEqual(exit_code, 0)
