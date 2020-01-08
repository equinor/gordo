import glob
from os.path import dirname, basename, isfile

# Loads all .py file in the current dir making them available for top level import
modules = glob.glob(dirname(__file__) + "/*.py")
__all__ = [
    basename(f)[:-3] for f in modules if isfile(f) and not f.endswith("__init__.py")
]
