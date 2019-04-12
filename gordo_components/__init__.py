from contextlib import redirect_stderr
import os

try:
    from ._version import version as __version__

    __version__ = __version__.replace("+", ".")
except ImportError:
    __version__ = "0.0.0"

# Hide the abundantly annoying 'Using X backend' from kears
with redirect_stderr(open(os.devnull, "w")):
    import keras  # noqa
