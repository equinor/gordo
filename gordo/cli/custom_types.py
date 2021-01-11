# -*- coding: utf-8 -*-
import ipaddress
import typing

import click


class HostIP(click.ParamType):
    """
    Validate input is a valid IP address
    """

    name = "host"

    def convert(self, value, param, ctx):
        try:
            ipaddress.ip_address(value)
            return value
        except ValueError as e:
            self.fail(e)


def key_value_par(val) -> typing.Tuple[str, str]:
    """
    Helpder for CLI input of 'key,val'
    """
    return val.split(",")
