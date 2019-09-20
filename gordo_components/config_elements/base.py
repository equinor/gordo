# -*- config: utf-8 -*-

import abc
from typing import Dict, Any


class ConfigElement(abc.ABC):
    """
    Represents functionality of an element found within a gordo config file.
    """

    @abc.abstractclassmethod
    def from_config(
        cls, config: Dict[str, Any], project_name: str, config_globals=None
    ) -> "ConfigElement":
        """
        Take one section of a config file and parse it into a given element
        """
        ...
