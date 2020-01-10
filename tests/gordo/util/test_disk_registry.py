# -*- coding: utf-8 -*-

import pathlib

from gordo.util import disk_registry


def test_simple_happy_path(tmpdir):
    """Tests that it works to write and read a simple value to a fresh registry"""
    disk_registry.write_key(tmpdir, "akey", "aval")
    assert disk_registry.get_value(tmpdir, "akey") == "aval"


def test_new_registry(tmpdir):
    """
    Tests that it works to write and read a simple value to a fresh registry with
    a non-existing directory
    """
    registry = pathlib.Path(tmpdir).joinpath("newregistry")
    disk_registry.write_key(registry, "akey", "aval")
    assert disk_registry.get_value(registry, "akey") == "aval"


def test_complicated_happy_path(tmpdir):
    """Tests that it works to write and read a 'complicated' value"""
    value = """
    A long
    value with many weird character lie åøæ
    and some linebreaks"""
    disk_registry.write_key(tmpdir, "akey", value)
    assert disk_registry.get_value(tmpdir, "akey") == value


def test_overwrites_existing(tmpdir):
    """Double writes to the same registry overwrites the first value"""
    the_key = "akey"
    first_value = "Some value"
    disk_registry.write_key(tmpdir, the_key, first_value)
    assert disk_registry.get_value(tmpdir, the_key) == first_value
    second_value = "Some value"
    disk_registry.write_key(tmpdir, the_key, second_value)
    assert disk_registry.get_value(tmpdir, the_key) == second_value


def test_delete(tmpdir):
    """Delete removes a key"""
    the_key = "akey"
    first_value = "Some value"
    disk_registry.write_key(tmpdir, the_key, first_value)
    assert disk_registry.get_value(tmpdir, the_key) == first_value

    existed_p = disk_registry.delete_value(tmpdir, the_key)
    assert disk_registry.get_value(tmpdir, the_key) is None
    # They key existed
    assert existed_p


def test_double_delete(tmpdir):
    """Delete works on non-existing key, returning False"""
    the_key = "akey"
    existed_p = disk_registry.delete_value(tmpdir, the_key)
    assert disk_registry.get_value(tmpdir, the_key) is None
    assert not existed_p


def test_get_value_without_registry_dir():
    """Test that it works to have registry_dir as None, and that it returns None as expected"""
    assert disk_registry.get_value(None, "akey") is None
