from typing import Tuple

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.0.0"


def _parse_version(version: str) -> Tuple[int, ...]:
    """
    Takes a string which starts with standard major.minor.patch.
    and returns the split of major minor and path as integers

    Parameters
    ----------
    version: str
        The semantic version string

    Returns
    -------
    Tuple[int, int, int]
        major, minor and patch versions
    """
    return tuple(int(i) for i in version.split(".")[:3])


MAJOR_VERSION, MINOR_VERSION, PATCH_VERSION = _parse_version(__version__)
