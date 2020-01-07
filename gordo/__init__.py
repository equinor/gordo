import logging
import os
import traceback
from typing import Tuple
import warnings

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.0.0"


def _parse_version(version: str) -> Tuple[int, ...]:
    """
    Takes a string which starts with standard major.minor.patch.
    and returns the split of major and minor version as integers

    Parameters
    ----------
    version: str
        The semantic version string

    Returns
    -------
    Tuple[int, int]
        major and minor versions
    """
    return tuple(int(i) for i in version.split(".")[:2])


MAJOR_VERSION, MINOR_VERSION = _parse_version(__version__)

try:
    # FIXME(https://github.com/abseil/abseil-py/issues/99)
    # FIXME(https://github.com/abseil/abseil-py/issues/102)
    # Unfortunately, many libraries that include absl (including Tensorflow)
    # will get bitten by double-logging due to absl's incorrect use of
    # the python logging library:
    #   2019-07-19 23:47:38,829 my_logger   779 : test
    #   I0719 23:47:38.829330 139904865122112 foo.py:63] test
    #   2019-07-19 23:47:38,829 my_logger   779 : test
    #   I0719 23:47:38.829469 139904865122112 foo.py:63] test
    # The code below fixes this double-logging.  FMI see:
    #   https://github.com/tensorflow/tensorflow/issues/26691#issuecomment-500369493

    import absl.logging

    logging.root.removeHandler(absl.logging._absl_handler)
    absl.logging._warn_preinit_stderr = False

except Exception:
    warnings.warn(f"Failed to fix absl logging bug {traceback.format_exc()}")
    pass
