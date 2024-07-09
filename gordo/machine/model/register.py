import logging
import inspect
from typing import Dict, Callable, Any  # pragma: no flakes
from gordo.machine.model.base import GordoBase
from tensorflow import keras

logger = logging.getLogger(__name__)


class register_model_builder:
    """
        Decorator to register a function as an available 'type' in supporting
        factory classes such as :class:`gordo.machine.model.models.KerasAutoEncoder`.

        When submitting the config file, it's important that the 'kind' is compatible
        with 'type'.

        ie. ``type='KerasAutoEncoder'`` should support the object returned by a given
        decorated function.


    .. code:: python

        from gordo_compontents.models.register import register_model_builder

        @register_model_builder(type='KerasAutoEncoder')
        def special_keras_model_builder(n_features, ...):
            ...

        A valid yaml config would be:
        model:
            gordo.machine.models.KerasAutoEncoder:
                kind: special_keras_model_builder

    """

    """
    Mapping of type: kind: function.
    ie.
    {
        'KerasAutoEncoder' : {'special_keras_model_builder': <gordo.builder.....>},
        'ScikitRandomForest': {'special_random_forest': <gordo.build....>}
    }
    """

    factories = dict()  # type: Dict[str, Dict[str, Callable[..., keras.models.Model]]]

    def __init__(self, type: str):
        self.type = type

    def __call__(self, model: Callable[..., keras.models.Model]):
        self._register(self.type, model)
        return model

    @classmethod
    def _register(cls, type: str, model: Callable[[int, Any], GordoBase]):
        """
        Registers a given function as an available factory under
        this type.
        """
        cls._validate_func(model)

        # Add function to available factories under this type
        if type not in cls.factories:
            cls.factories[type] = dict()
        cls.factories[type][model.__name__] = model

    @staticmethod
    def _validate_func(func):
        # Any validity checks before registering their function
        if "n_features" not in inspect.getfullargspec(func).args:
            raise ValueError(
                f"Build function: {func.__name__} does not have "
                "'n_features' as an argument; it should."
            )
