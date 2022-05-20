import pytest

from gordo.builder.build_model import ModelBuilder
from gordo.builder.utils import create_model_builder


def test_create_model_builder_success():
    cls = create_model_builder(None)
    assert cls is ModelBuilder
    cls = create_model_builder("gordo.builder.build_model.ModelBuilder")
    assert cls is ModelBuilder


def test_create_model_builder_failed():
    with pytest.raises(ImportError):
        create_model_builder("wrong.import.WrongClass")
    with pytest.raises(
        ValueError,
        match='"Machine" class located in "gordo.machine.Machine" should be subclass of "ModelBuilder"',
    ):
        create_model_builder("gordo.machine.Machine")
