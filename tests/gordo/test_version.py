# -*- coding: utf-8 -*-

import pytest
from typing import Tuple

from gordo import parse_version, __version__


def test_version():
    assert isinstance(__version__, str)
    major, minor, is_unstable = parse_version(__version__)
    assert isinstance(major, int)
    assert isinstance(minor, int)
    assert isinstance(is_unstable, bool)


@pytest.mark.parametrize(
    "version,expected",
    [
        ("1.1.1", (1, 1, False)),
        ("1.1.1.dev-a1", (1, 1, True)),
        ("0.55.0-rc1", (0, 55, True)),
        ("0.0.0", (0, 0, True)),
    ],
)
def test_version_parser(version: str, expected: Tuple[int, int, int]):
    assert parse_version(version) == expected


def test_version_with_error():
    with pytest.raises(ValueError):
        parse_version("not_a_version")
