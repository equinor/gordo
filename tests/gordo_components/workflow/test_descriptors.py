# -*- coding: utf-8 -*-

import unittest
import datetime

from gordo_components.workflow.config_elements.validators import (
    ValidUrlString,
    ValidDatetime,
    ValidTagList,
    ValidMetadata,
    ValidModel,
    ValidMachineRuntime,
    fix_resource_limits,
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

        myclass.value = datetime.datetime.now(tz=datetime.timezone.utc)
        # Should raise an error
        with self.assertRaises(ValueError):
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

    def test_valid_runtime(self):
        """
        Verifies that ValidRuntime enforces a server section with resources.requests and
        resources.limits
        """

        class MyClass:
            value = ValidMachineRuntime()

        myclass = MyClass()

        # Runtimes should be dicts
        myclass.value = {"influx": {"enable": True}}

        # Problem.
        with self.assertRaises(ValueError):
            myclass.value = 2

    def test_fix_resource_limits(self):
        # Bumps cpu limit
        resource_dict = {
            "requests": {"memory": 1, "cpu": 5},
            "limits": {"memory": 3, "cpu": 4},
        }
        res = fix_resource_limits(resource_dict)
        self.assertEqual(res["limits"]["cpu"], 5)

        # Works without requests
        resource_dict = {"limits": {"memory": 3, "cpu": 4}}
        res = fix_resource_limits(resource_dict)
        self.assertEqual(res["limits"]["cpu"], 4)
        # Works without limits
        resource_dict = {"requests": {"memory": 1, "cpu": 5}}
        res = fix_resource_limits(resource_dict)
        self.assertEqual(res["requests"]["cpu"], 5)

        # Does  nothing if limit>requests
        resource_dict = {
            "requests": {"memory": 1, "cpu": 5},
            "limits": {"memory": 3, "cpu": 6},
        }
        res = fix_resource_limits(resource_dict)
        self.assertEqual(res["limits"]["cpu"], 6)

    def test_fix_resource_limits_checks_ints(self):
        resource_dict = {
            "requests": {"memory": "1M", "cpu": 5},
            "limits": {"memory": 3, "cpu": 4},
        }
        with self.assertRaises(ValueError):
            fix_resource_limits(resource_dict)
