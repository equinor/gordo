from subprocess import CompletedProcess

import pytest
from mock import patch

from gordo.workflow.workflow_generator.helpers import determine_argo_version, ArgoVersionDeterminingError

_argo_command = ["argo", "version", "--short"]


def create_completed_process(return_code, stdout):
    return CompletedProcess(_argo_command, return_code, stdout=stdout, stderr=None)


def test_determine_argo_version_success():
    completed_process = create_completed_process(0, b"argo: v1.1.1\n")
    with patch("subprocess.run", return_value=completed_process):
        version = determine_argo_version()
        assert str(version) == "1.1.1"


def test_determine_argo_version_fail():
    with patch("subprocess.run", side_effect=FileNotFoundError("argo")):
        with pytest.raises(ArgoVersionDeterminingError):
            determine_argo_version()
    with patch("subprocess.run", return_value=create_completed_process(0, b"\xa0\xa1")):
        with pytest.raises(ArgoVersionDeterminingError):
            determine_argo_version()
    with patch("subprocess.run", return_value=create_completed_process(0, b"wrong output")):
        with pytest.raises(ArgoVersionDeterminingError):
            determine_argo_version()
