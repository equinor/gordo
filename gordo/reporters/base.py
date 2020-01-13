import abc
import pydoc

from typing import Dict, Any

from gordo.machine import Machine


class BaseReporter(abc.ABC):
    @abc.abstractmethod
    def report(self, machine: Machine):
        """Report/log the machine"""

    def to_dict(self) -> dict:
        """
        Serialize this object into a dict representation, which can be used to
        initialize a new object after popping 'type' from the dict.

        Returns
        -------
        dict
        """
        if not hasattr(self, "_params"):
            raise AttributeError(
                f"Failed to lookup init parameters, ensure the "
                f"object's __init__ is decorated with 'capture_args'"
            )
        # Update dict with the class
        params = self._params  #  type: ignore
        for key, value in params.items():
            if hasattr(value, "to_dict"):
                params[key] = value.to_dict()

        import_path = f"{self.__module__}.{self.__class__.__name__}"
        return {import_path: params}

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "BaseReporter":
        """
        Reconstruct the reporter from a dict representation or a single
        import path if it doesn't require any init parameters.
        """
        if isinstance(config, dict):
            keys = list(config.keys())
            if len(keys) != 1:
                raise ValueError(
                    "If a dict, the reporter should have a single key as its import path "
                    "mapped to the init parameters for it."
                )
            import_path = keys[0]
        else:
            import_path = config  # type: ignore

        Reporter = pydoc.locate(import_path)
        if Reporter is None:
            raise LookupError(f"Could not find reporter: {import_path}")

        if isinstance(config, dict):
            return Reporter(**config[import_path])  # type: ignore
        else:
            return Reporter()  # type: ignore
