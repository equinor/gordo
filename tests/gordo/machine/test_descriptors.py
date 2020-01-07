# -*- coding: utf-8 -*-

import datetime

import pytest

from gordo.machine.validators import (
    ValidUrlString,
    ValidDatetime,
    ValidTagList,
    ValidMetadata,
    ValidModel,
    ValidMachineRuntime,
    fix_resource_limits,
)


def test_valid_model():
    """
    Verifies a given object is Union[str, dict]
    """

    class MyClass:
        value = ValidModel()

    myclass = MyClass()

    # All ok
    myclass.value = {
        "gordo.machine.model.models.KerasAutoEncoder": {"kind": "feedforward_hourglass"}
    }
    myclass.value = "sklearn.ensemble.forest.RandomForestRegressor"

    # Not ok
    with pytest.raises(ValueError):
        myclass.value = 1
    with pytest.raises(ValueError):
        myclass.value = None


def test_valid_metadata():
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
    with pytest.raises(ValueError):
        myclass.value = 1
    with pytest.raises(ValueError):
        myclass.value = "string"


def test_valid_datetime():
    """
    Verifies a given object is of datetime.datetime
    """

    class MyClass:
        value = ValidDatetime()

    myclass = MyClass()

    myclass.value = datetime.datetime.now(tz=datetime.timezone.utc)
    # Should raise an error
    with pytest.raises(ValueError):
        myclass.value = datetime.datetime.now()

    # Should raise an error
    with pytest.raises(ValueError):
        myclass.value = "not a datetime object"


def test_valid_tag_list():
    """
    Verifies a given object is of a non-empty List[str]
    """

    class MyClass:
        value = ValidTagList()

    myclass = MyClass()

    # No problem
    myclass.value = ["string here", "string there"]

    # Problem.
    with pytest.raises(ValueError):
        myclass.value = "not a list"
    with pytest.raises(ValueError):
        myclass.value = []  # Error with empty list.


def test_valid_url_string():
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
        assert ValidUrlString.valid_url_string(
            valid_name
        ), f"Expected '{valid_name}' to be valid, but it is not"

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
        assert not ValidUrlString.valid_url_string(
            invalid_name
        ), f"Expected '{invalid_name}' to be invalid, but it is."

    # Test it's a working Descriptor
    class MyClass:
        value = ValidUrlString()

    myclass = MyClass()

    myclass.value = "this-should-not-cause-an-error"

    with pytest.raises(ValueError):
        myclass.value = (
            "this-name-is-simply-too-long-that-is-the-only-problem-here-for-sure"
        )

    with pytest.raises(ValueError):
        myclass.value = "but this should!"


def test_valid_runtime():
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
    with pytest.raises(ValueError):
        myclass.value = 2


def test_fix_resource_limits():
    # Bumps cpu limit
    resource_dict = {
        "requests": {"memory": 1, "cpu": 5},
        "limits": {"memory": 3, "cpu": 4},
    }
    res = fix_resource_limits(resource_dict)
    assert res["limits"]["cpu"] == 5

    # Works without requests
    resource_dict = {"limits": {"memory": 3, "cpu": 4}}
    res = fix_resource_limits(resource_dict)
    assert res["limits"]["cpu"] == 4
    # Works without limits
    resource_dict = {"requests": {"memory": 1, "cpu": 5}}
    res = fix_resource_limits(resource_dict)
    assert res["requests"]["cpu"] == 5

    # Does  nothing if limit>requests
    resource_dict = {
        "requests": {"memory": 1, "cpu": 5},
        "limits": {"memory": 3, "cpu": 6},
    }
    res = fix_resource_limits(resource_dict)
    assert res["limits"]["cpu"] == 6


def test_fix_resource_limits_checks_ints():
    resource_dict = {
        "requests": {"memory": "1M", "cpu": 5},
        "limits": {"memory": 3, "cpu": 4},
    }
    with pytest.raises(ValueError):
        fix_resource_limits(resource_dict)
