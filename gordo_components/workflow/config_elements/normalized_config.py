# -*- coding: utf-8 -*-

from typing import List

from .machine import Machine


class NormalizedConfig:
    """
    Represents a fully loaded config file
    """

    machines: List[Machine]

    def __init__(self, machines: List[Machine]):
        self.machines = machines

    @classmethod
    def from_config(cls, config: dict) -> "NormalizedConfig":
        """
        Create a normalized config from one which can contain 'globals'
        """
        global_config_vars = config.get("globals", dict())
        machines = [
            Machine.from_config(conf, config_globals=global_config_vars)
            for conf in config["machines"]
        ]  # type: List[Machine]
        return cls(machines)
