# -*- config: utf-8 -*-

import abc
from typing import Dict


class ConfigElement(abc.ABC):
    """
    Represents functionality of an element found within a gordo config file.
    """

    @abc.abstractclassmethod
    def from_config(cls, config: Dict[str, dict]) -> "ConfigElement":
        """
        Take one section of a config file and parse it into a given element
        """
        ...
