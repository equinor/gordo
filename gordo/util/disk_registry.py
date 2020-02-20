import os
from pathlib import Path
from typing import Union, AnyStr, Optional

import logging

logger = logging.getLogger(__name__)

"""
A simple file-based key/value registry. Each key gets a file with filename = key, and
the content of the file is the value. No fancy. Why? Simple, and there is no problems
with concurrent writes to different keys. Concurrent writes to the same key will break 
stuff. 
"""


def write_key(registry_dir: Union[os.PathLike, str], key: str, val: AnyStr):
    """ Registers a key-value combination into the register. Key must valid as a
    filename.

    Parameters
    ----------
    registry_dir: Union[os.PathLike, str]
        Path to the registry. If it does not exists it will be created, including any
        missing folders in the path.
    key: str
        Key to use for the key/value. Must be valid as a filename.
    val: AnyStr
        Value to write to the registry.

    Examples
    --------
    In the following example we use the temp directory as the registry
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     write_key(tmpdir, "akey", "aval")
    ...     get_value(tmpdir, "akey")
    'aval'
    """
    key_file_path = Path(registry_dir).joinpath(key)
    logger.info(f"Storing object with key {key} to {key_file_path}")
    # If the model location already exists in the cache we overwrite it
    if key_file_path.exists():
        logger.warning(f"Key {key} already exists, " f"overwriting")
    elif not Path(registry_dir).exists():
        logger.debug(
            f"Found that registry dir {registry_dir} does not exists, creating it"
        )
        # If someone else creates the directory at the same time we accept that by
        # exists_ok=True
        os.makedirs(registry_dir, exist_ok=True)
    with key_file_path.open(mode="w") as f:
        f.write(val)  # type: ignore


def get_value(registry_dir: Union[os.PathLike, str], key: str) -> Optional[AnyStr]:
    """
    Retrieves the value with key reg_key from the registry, None if it does not
    exists.

    Parameters
    ----------
    registry_dir: Union[os.PathLike, str]
        Path to the registry. If it does not exist we return None
    key: str
        Key to look up in the registry.

    Returns
    -------
    Optional[AnyStr]:
        The value of `key` in the registry, None if no value is registered with that key
        in the registry.
    """
    output_val = None

    if registry_dir is None:
        return output_val

    key_file_path = Path(registry_dir).joinpath(key)
    logger.info(f"Looking for registry key {key} at path " f"{key_file_path}")
    # If the model location exists
    if key_file_path.exists():
        with key_file_path.open(mode="r") as f:
            output_val = f.read()
        logger.debug(f"Read value {output_val} from registry")
    else:
        logger.info(f"Did not find registry key {key} at path {key_file_path}")
    return output_val  # type: ignore


def delete_value(registry_dir: Union[os.PathLike, str], key: str) -> bool:
    """
    Deletes the value with key reg_key from the registry, and returns True if it
    existed.

    Parameters
    ----------
    registry_dir: Union[os.PathLike, str]
        Path to the registry. Does not need to exist
    key: str
        Key to look up in the registry.

    Returns
    -------
    bool:
        True if the key existed, false otherwise
    """
    key_file_path = Path(registry_dir).joinpath(key)
    logger.info(f"Looking for registry key {key} at path " f"{key_file_path}")
    # If the model location exists
    if key_file_path.exists():
        key_file_path.unlink()
        logger.debug(f"Removed key {key} from registry")
        return True
    else:
        logger.info(f"Did not find registry key {key} at path {key_file_path}")
    return False
