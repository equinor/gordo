import abc

from typing import Dict, Any

from gordo.machine import Machine
from gordo import serializer


class BaseReporter(abc.ABC):
    @abc.abstractmethod
    def report(self, machine: Machine):
        """Report/log the machine"""

    def get_params(self, deep=False):
        return self._params.copy()

    def to_dict(self) -> dict:
        """
        Serialize this object into a dict representation, which can be used to
        initialize a new object after popping 'type' from the dict.

        Returns
        -------
        dict
        """
        return serializer.into_definition(self)

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "BaseReporter":
        """
        Reconstruct the reporter from a dict representation or a single
        import path if it doesn't require any init parameters.
        """
        return serializer.from_definition(config)
