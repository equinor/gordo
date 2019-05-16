# -*- coding: utf-8 -*-

import os
import json
import ast
import typing

import click
from dateutil import parser

from gordo_components.data_provider import providers


class DataProviderParam(click.ParamType):
    """
    Load a DataProvider from JSON/dict representation or from a JSON file
    """

    name = "data-provider"

    def convert(self, value, param, ctx):
        if os.path.isfile(value):
            with open(value) as f:
                kwargs = json.load(f)
        else:
            kwargs = ast.literal_eval(value)

        if "type" not in kwargs:
            self.fail(f"Cannot create DataProvider without 'type' key defined")

        kind = kwargs.pop("type")

        Provider = getattr(providers, kind, None)
        if Provider is None:
            self.fail(f"No DataProvider named '{kind}'")
        return Provider(**kwargs)


class IsoFormatDateTime(click.ParamType):
    """
    Parse a string into an ISO formatted datetime object
    """

    name = "iso-datetime"

    def convert(self, value, param, ctx):
        try:
            return parser.isoparse(value)
        except ValueError:
            self.fail(f"Failed to parse date '{value}' as ISO formatted date'")


def key_value_par(val) -> typing.Tuple[str, str]:
    """
    Helpder for CLI input of 'key,val'
    """
    return val.split(",")
