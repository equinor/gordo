import pytest

from gordo.util.version import (
    parse_version,
    GordoRelease,
    GordoSpecial,
    Special,
    GordoPR,
    GordoSHA,
)


def test_release():
    release = GordoRelease(1, 2, 3)
    assert not release.only_major()
    assert not release.only_major_minor()
    assert not release.without_patch()
    release = GordoRelease(1, 2)
    assert not release.only_major()
    assert release.only_major_minor()
    assert release.without_patch()
    release = GordoRelease(1)
    assert release.only_major()
    assert not release.only_major_minor()
    assert release.without_patch()


@pytest.mark.parametrize(
    "gordo_version,expected",
    [
        ("1.2.3", GordoRelease(1, 2, 3)),
        ("3.4.5dev2", GordoRelease(3, 4, 5, "dev2")),
        ("5.7", GordoRelease(5, 7)),
        ("latest", GordoSpecial(Special.LATEST)),
        ("pr-43", GordoPR(43)),
        ("dke0832k", GordoSHA("dke0832k")),
    ],
)
def test_versions(gordo_version, expected):
    result = parse_version(gordo_version)
    assert result == expected
    assert result.get_version() == gordo_version


def test_exceptions():
    with pytest.raises(ValueError):
        parse_version("pr-2d")
    with pytest.raises(ValueError):
        parse_version("0-1")
