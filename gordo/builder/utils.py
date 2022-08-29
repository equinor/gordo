from typing import Optional, Type

from gordo_core.import_utils import import_location

from .build_model import ModelBuilder


def create_model_builder(model_builder_class: Optional[str]) -> Type[ModelBuilder]:
    if model_builder_class is None:
        return ModelBuilder
    cls = import_location(model_builder_class)
    if not issubclass(cls, ModelBuilder):
        raise ValueError(
            '"%s" class located in "%s" should be subclass of "%s"'
            % (cls.__name__, model_builder_class, ModelBuilder.__name__)
        )
    return cls
