from typing import Optional, Type

from gordo.serializer.utils import validate_locate, import_locate

from .build_model import ModelBuilder


def create_model_builder(model_builder_class: Optional[str]) -> Type[ModelBuilder]:
    if model_builder_class is None:
        return ModelBuilder
    validate_locate(model_builder_class)
    cls = import_locate(model_builder_class)
    if issubclass(cls, ModelBuilder):
        raise ValueError('"%s" class should be subclass of "%s"')
    return cls
