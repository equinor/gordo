import pytest
import posixpath

from unittest.mock import MagicMock


@pytest.fixture
def mock_file_system():
    mock = MagicMock()
    mock.join.side_effect = posixpath.join
    mock.split.side_effect = posixpath.split
    return mock
