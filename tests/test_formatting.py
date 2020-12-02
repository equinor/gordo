# -*- coding: utf-8 -*-
import os
import sys


def test_formatting_black():
    project_path = os.path.join(os.path.dirname(__file__), "..")
    gordo_path = os.path.join(project_path, "gordo")
    tests_path = os.path.join(project_path, "tests")
    cmd = [
        sys.executable,
        "-m",
        "black",
        "--check",
        "-v",
        gordo_path,
        tests_path,
        "--exclude",
        r".*_version.py",
    ]
    exit_code = os.system(" ".join(cmd))
    assert exit_code == 0
