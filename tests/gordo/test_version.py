# -*- coding: utf-8 -*-

import pytest
from typing import Tuple

from gordo import _parse_version, __version__


def test_version():
    assert isinstance(__version__, str)
    versions: (int, int, int) = _parse_version(__version__)
    for v in versions:
        assert isinstance(v, int)


@pytest.mark.parametrize(
    "version,expected",
    [
        ("1.1.1", (1, 1)),
        ("1.1.1.dev-a1", (1, 1)),
        ("0.55.25.02", (0, 55)),
        ("0.0.0", (0, 0)),
    ],
)
def test_version_parser(version: str, expected: Tuple[int, int, int]):
    assert _parse_version(version) == expected
