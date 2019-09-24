from typing import Any

import pytest
import threading

from gordo_components.model.metadata import MetadataCollector


@pytest.mark.parametrize(
    "data,input_key,expected_key", [({"a": 0}, "a", "a_1"), ({"a_1": 1}, "a_1", "a_2")]
)
def test_increment_valid_input(data, input_key, expected_key):
    """
    Test that increment method increments a string suffix as expected
    """
    assert MetadataCollector.increment(data, input_key) == expected_key


@pytest.mark.parametrize("input_key", [(None), (42.0), (int)])
def test_increment_invalid_input(input_key):
    """
    Test that an exception is raised if invalid keys are passed to `increment`
    """
    d = {}
    with pytest.raises(ValueError):
        MetadataCollector.increment(d, input_key)


@pytest.mark.parametrize(
    "log_meta_kwargs,multiple_created,expected_first,expected_second",
    [
        # 1. Default: Create multiple entries
        ({}, True, 0, 1),
        # 2. Overwrite
        ({"overwrite": True}, False, 1, None),
        # 3. No multiples
        ({"no_multiples": True}, False, 0, None),
        # 4. Overwrite and no multiples
        ({"no_multiples": True, "overwrite": True}, False, 1, None),
    ],
)
def test_metadatacollector_modes(
    log_meta_kwargs: dict,
    multiple_created: bool,
    expected_first: Any,
    expected_second: Any,
):
    """
    Test the different operating modes of the MetadataCollector class

    1. Default: Multiple entry created, both entries have different values
    2. Overwrite: No multiple entry created, value matches second entered
    3. No multiples: No multiple entry created, value matches first entered
    4. Overwrite and no multiples: No multiple entry created, value matches second entered

    Parameters
    ----------
    log_meta_kwargs: dict
        The kwargs to pass to MetadataColector
    multiple_created: bool
        True if a multiple should be created with given MetadataCollector kwargs
    expected_second: Any
        The value expected for the second entry of `test_key`
    expected_second: Any
        The value expected for the second entry of `test_key`

    Attributes
    ----------
    input_first: Any
        The value set for the first entry of `test_key`
    input_second: Any
        The value set for the second entry of `test_key`
    """

    test_key = "test_key"
    multiple_key = "test_key_1"

    def func():
        """Use the log_meta method for two logging calls with the same key"""
        MetadataCollector.log_meta(test_key, 0, **log_meta_kwargs)
        MetadataCollector.log_meta(test_key, 1, **log_meta_kwargs)

    with MetadataCollector() as mc:
        # Run code that has calls to `MetadataCollector.log_meta`
        func()

        # Check that the first entry value is as expected
        assert mc.metadata[test_key] == expected_first

        if multiple_created:
            # Check that the value of the existing key is as expected
            assert mc.metadata[multiple_key] == expected_second
        else:
            # Check that a new entry was not made if not expected
            assert multiple_key not in mc.metadata


def test_threadlocal_handling():
    """
    Test that local variable attribute data garbage collected in __exit___

    Metadata extracted should still be available
    """

    test_key = "test_key"

    with MetadataCollector() as mc:

        # Check that MetadataCollector thread local attribute has _gordo_metadata
        assert hasattr(mc._THREAD_LOCAL, "_gordo_metadata")

        MetadataCollector.log_meta(test_key, 0)
        metadata = mc.metadata

    # Check that threading local data removed after context manager
    tls = threading.local()
    assert not hasattr(tls, "_gordo_metadata")

    # Check that the copy exists after context manager
    assert test_key in metadata
