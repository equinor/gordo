# -*- coding: utf-8 -*-

import tempfile
import pathlib

from gordo_components.util import disk_registry


def test_simple_happy_path():
    """Tests that it works to write and read a simple value to a fresh registry"""
    with tempfile.TemporaryDirectory() as tmpdir:
        disk_registry.write_key(tmpdir, "akey", "aval")
        assert disk_registry.get_value(tmpdir, "akey") == "aval"


def test_new_registry():
    """Tests that it works to write and read a simple value to a fresh registry with
     a non-existing directory"""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = pathlib.Path(tmpdir).joinpath("newregistry")
        disk_registry.write_key(registry, "akey", "aval")
        assert disk_registry.get_value(registry, "akey") == "aval"


def test_complicated_happy_path():
    """Tests that it works to write and read a 'complicated' value"""
    with tempfile.TemporaryDirectory() as tmpdir:
        value = """
        A long
        value with many weird character lie åøæ
        and some linebreaks"""
        disk_registry.write_key(tmpdir, "akey", value)
        assert disk_registry.get_value(tmpdir, "akey") == value


def test_overwrites_existing():
    """Double writes to the same registry overwrites the first value"""
    the_key = "akey"
    with tempfile.TemporaryDirectory() as tmpdir:
        first_value = "Some value"
        disk_registry.write_key(tmpdir, the_key, first_value)
        assert disk_registry.get_value(tmpdir, the_key) == first_value
        second_value = "Some value"
        disk_registry.write_key(tmpdir, the_key, second_value)
        assert disk_registry.get_value(tmpdir, the_key) == second_value
