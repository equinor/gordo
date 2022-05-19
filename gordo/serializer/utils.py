import importlib
import re

from typing import Any

import_path_item_re = re.compile(r"^[a-z\d_-]+$", re.IGNORECASE)


def validate_locate(import_path: str):
    """
    Validate python import path.

    Parameters
    ----------
    import_path: str

    Returns
    -------

    """
    import_path_items = import_path.split(".")
    for item in import_path_items:
        if not import_path_item_re.match(item):
            raise ValueError('Invalid import path "%s" item "%s"' % (import_path, item))


def import_locate(import_path: str) -> Any:
    """
    Import entity from provided `import_path`.

    Example
    -------
    >>> import_locate("sklearn.pipeline.Pipeline")
    <class 'sklearn.pipeline.Pipeline'>

    Parameters
    ----------
    import_path: str

    Returns
    -------
    Imported entity.

    """
    try:
        module_name, class_name = import_path.rsplit(".", 1)
    except ValueError:
        raise ImportError("Malformed import path: %s" % import_path)
    module = importlib.import_module(module_name)
    if not hasattr(module, class_name):
        raise ImportError(
            "Unable to find class %s in module %s" % (module_name, class_name)
        )
    cls = getattr(module, class_name)
    return cls
