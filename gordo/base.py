from typing import Dict, Any
from gordo import serializer


class GordoBase:
    """
    Base object for all other Gordo elements.
    Aid in de/serializing them from and into primitive representations.

    Must define __init__ with @capture_args decorator
    """

    _params: Dict[str, Any] = dict()

    def get_params(self, deep=False):
        """
        Get the parameters this object was initialized with
        """
        return {key: getattr(self, key, value) for key, value in self._params.items()}

    def to_dict(self) -> dict:
        """
        Serialize this object into a dict representation, suitable to be
        loaded by :func:`gordo.base.GordoBase.from_dict`

        Returns
        -------
        dict
        """
        return serializer.into_definition(self)

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "GordoBase":
        """
        Reconstruct the object from the output of :func:`gordo.base.GordoBase.to_dict`
        """
        return serializer.from_definition(config)
