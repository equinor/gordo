from typing import Any, Optional

import json
import logging
import threading

logger = logging.getLogger(__name__)


class MetadataCollector:
    """
    Theaded metadata logging class

    Attributes
    ----------
    _tls: theading.local
        Variable with associated thread.
    _tls._gordo_metadata: dict
        Attribute of local variable to store metadata to.
    metadata: dict
        A helper reference to `_tls._gordo_metadata` for access under context manager.

    Examples
    --------
    >>> def func():
    ...     MetadataCollector.log_meta("test_key", 42)
    ...     return
    >>> with MetadataCollector() as mc:
    ...     func()
    ...     print(mc.metadata["test_key"])
    42
    """

    _THREAD_LOCAL = threading.local()

    def __enter__(self):
        self._THREAD_LOCAL._gordo_metadata = {}
        self.metadata = self._THREAD_LOCAL._gordo_metadata
        return self

    def __exit__(self, exc_type, exc_value, tb):
        """
        Remove collecter object for Thread cleanliness
        """
        del self._THREAD_LOCAL._gordo_metadata

    def get_metadata(self):
        """
        Return the metadata dictionary from the threading local variable
        """
        return self._THREAD_LOCAL._gordo_metadata

    @classmethod
    def log_meta(
        cls,
        key: str,
        value: Any,
        no_multiples: Optional[bool] = False,
        overwrite: Optional[bool] = False,
    ):
        """
        Log key/value pair to gordo thread local variable

        The default behavior is to create multiple entries for a given key, where
        the keys for subsequently added entries are appended with an increasing
        integer suffix (E.g.  `example`, `example_1`, `example_2`, etc.).

        When `no_multiples` is set and the key already exists in the thread local
        metadata, logging will be skipped and a warning will be logged.

        When `overwrite` is set and the key already exists in the thread local
        metadata, the existing value will be overwritten and no new entry will be made.
        Setting `no_multiples` when `overwrite is set has no effect.

        Parameters
        ----------
        key: str
            The key to add to the metadata dictionary.
        value: Any
            The value to add to the metadata dictionary.
        no_multiples: Optional[bool]
            Flag to not allow multiple values for a single key.
        overwrite: Optional[bool]
            Flag to allow overwriting of values for keys that already exist.
        """

        if not is_json_serializable(value):
            raise ValueError("Value with type {type(value}} must be JSON serializable.")

        if hasattr(cls._THREAD_LOCAL, "_gordo_metadata"):
            metadata = cls._THREAD_LOCAL._gordo_metadata
            if key in metadata:
                if overwrite:
                    logger.warning(
                        f"Value for key {key} was found and has been overwritten."
                    )
                    metadata.update({key: value})
                elif no_multiples:
                    logger.warning(
                        f"Value for key {key} was found and an additional entry will not be written."
                    )
                else:
                    metadata.update({MetadataCollector.increment(metadata, key): value})
            else:
                metadata.update({key: value})

    @staticmethod
    def increment(d: dict, key: str) -> str:
        """
        Add sequential suffix if key exists

        Parameters
        ----------
        d: dict
            Dictionary to check for existing key.
        key: str
            Key to check if exists in dictionary.

        Returns
        -------
        key: str
            Key with integer suffix added.

        Examples
        --------
        >>> d = {"test_b":0, "test_c_1":0}
        >>> print(MetadataCollector.increment(d, "test_a"))
        test_a
        >>> print(MetadataCollector.increment(d, "test_b"))
        test_b_1
        >>> print(MetadataCollector.increment(d, "test_c"))
        test_c
        >>> print(MetadataCollector.increment(d, "test_c_1"))
        test_c_2
        """
        if type(key) is not str:
            raise ValueError(f"Key {key} must be of string type, not {type(key)}")

        splits = key.split("_")
        key_base = ""
        try:
            i = int(splits[-1])
            key_base = f"{'_'.join(splits[:-1])}"
        except ValueError:
            i = 0
            key_base = key
        except Exception as e:
            raise e

        if (key in d) or (key_base) in d:
            key = f"{key_base}_{i+1}"

        return key


def is_json_serializable(obj: Any) -> bool:
    """
    Check that an object can be JSON serialized

    Parameters
    ----------
    obj: Any
        Object to check.

    Returns
    -------
    bool:
        Flag indication if object can be serialized (True) or not (False).

    Examples
    --------
    >>> is_json_serializable({"key1": 5, "key2": [1,2,3]})
    True
    >>> class TestClass():
    ...     '''Needs a __json__ method'''
    ...     def TestFunc():
    ...         return
    >>> is_json_serializable(TestClass())
    False
    """
    try:
        json.dumps(obj)
        return True
    except:
        return False
