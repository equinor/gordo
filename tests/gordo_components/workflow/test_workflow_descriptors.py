# -*- coding: utf-8 -*-

import unittest
import datetime

from gordo_components.workflow.config_elements.validators import (
    ValidUrlString,
    ValidDatetime,
    ValidTagList,
    ValidMetadata,
    ValidModel,
)


class DescriptorsTestCase(unittest.TestCase):
    def test_valid_model(self):
        """
        Verifies a given object is Union[str, dict]
        """

        class MyClass:
            value = ValidModel()

        myclass = MyClass()

        # All ok
        myclass.value = {"KerasModel": {"kind": "symetric"}}
        myclass.value = "KerasModel"

        # Not ok
        with self.assertRaises(ValueError):
            myclass.value = 1
        with self.assertRaises(ValueError):
            myclass.value = None

    def test_valid_metadata(self):
        """
        Verifies a given object is Optional[dict]
        """

        class MyClass:
            value = ValidMetadata()

        myclass = MyClass()

        # All of these are ok
        myclass.value = dict()
        myclass.value = {"key": "value"}
        myclass.value = None

        # This is not
        with self.assertRaises(ValueError):
            myclass.value = 1
        with self.assertRaises(ValueError):
            myclass.value = "string"

    def test_valid_datetime(self):
        """
        Verifies a given object is of datetime.datetime
        """

        class MyClass:
            value = ValidDatetime()

        myclass = MyClass()

        # Should be a-ok.
        myclass.value = datetime.datetime.now()

        # Should raise an error
        with self.assertRaises(ValueError):
            myclass.value = "not a datetime object"

    def test_valid_tag_list(self):
        """
        Verifies a given object is of a non-empty List[str]
        """

        class MyClass:
            value = ValidTagList()

        myclass = MyClass()

        # No problem
        myclass.value = ["string here", "string there"]

        # Problem.
        with self.assertRaises(ValueError):
            myclass.value = "not a list"
        with self.assertRaises(ValueError):
            myclass.value = []  # Error with empty list.

    def test_valid_url_string(self):
        """
        Verifies that a given string matches what we consider to be a valid
        url; in this case alphanumeric with dashes.
        """
        valid_names = [
            "valid-name-here",
            "validnamehere",
            "also-a-valid-name123",
            "equally-valid-name",
            "another-1-2-3",
        ]
        for valid_name in valid_names:
            self.assertTrue(
                ValidUrlString.valid_url_string(valid_name),
                msg=f"Expected '{valid_name}' to be valid, but it is not",
            )

        invalid_names = [
            "Not_a_valid_name",
            "C%tainly-not-v@lid",
            "also not valid with spaces",
            "couldn't-possibly-valid",
            "also,-this-is-not-valid",
            "(this-is-not-either)",
            "!nor-this",
            "cannot-have-UpperCase",
            "-cannot-begin-with-dashes",
            "cannot-end-with-dashes-",
        ]
        for invalid_name in invalid_names:
            self.assertFalse(
                ValidUrlString.valid_url_string(invalid_name),
                msg=f"Expected '{invalid_name}' to be invalid, but it is.",
            )

        # Test it's a working Descriptor
        class MyClass:
            value = ValidUrlString()

        myclass = MyClass()

        myclass.value = "this-should-not-cause-an-error"

        with self.assertRaises(ValueError):
            myclass.value = (
                "this-name-is-simply-too-long-that-is-the-only-problem-here-for-sure"
            )

        with self.assertRaises(ValueError):
            myclass.value = "but this should!"
