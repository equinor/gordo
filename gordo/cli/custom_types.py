# -*- coding: utf-8 -*-
import ipaddress
import re

import click
import json

from pydantic import parse_obj_as, ValidationError
from typing import Tuple, TypeVar, Generic, Type, Optional, Any

T = TypeVar("T")


class JSONParam(click.ParamType, Generic[T]):
    """
    Loads JSON and validates value against pydantic schema
    """

    name = "JSON"

    def __init__(self, schema: Type[T]):
        self.schema = schema

    def convert(
        self, value: Any, param: Optional[click.Parameter], ctx: Optional[click.Context]
    ) -> Optional[T]:
        if value is None:
            return None
        try:
            data = json.loads(value)
        except json.JSONDecodeError as e:
            self.fail("Malformed JSON string - %s" % str(e))
        try:
            obj = parse_obj_as(self.schema, data)
        except ValidationError as e:
            self.fail("Schema validation error - %s" % str(e))
        return obj


class REParam(click.ParamType):
    """
    Validates argument over a regular expression
    """

    name = "REGEXP"

    def __init__(self, pattern: str):
        self.pattern = pattern
        self.re = re.compile(pattern)

    def convert(
        self, value: Any, param: Optional[click.Parameter], ctx: Optional[click.Context]
    ):
        m = self.re.match(value)
        if not m:
            self.fail("Value '%s' not match '%s'" % (value, self.pattern))
        return value


class HostIP(click.ParamType):
    """
    Validate input is a valid IP address
    """

    name = "host"

    def convert(
        self, value: Any, param: Optional[click.Parameter], ctx: Optional[click.Context]
    ):
        try:
            ipaddress.ip_address(value)
            return value
        except ValueError as e:
            self.fail(str(e))


def key_value_par(val) -> Tuple[str, str]:
    """
    Helper for CLI input of 'key,val'
    """
    return val.split(",")
