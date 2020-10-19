import pytest

from io import StringIO
from gordo.machine.dataset.data_provider.assets_config import (
    AssetsConfig,
    PathSpec,
    ConfigException,
    validation_error_exception_message,
    exception_message,
)
from gordo.machine.dataset.data_provider.resource_assets_config import (
    load_assets_config,
)
from marshmallow import ValidationError


succeeded_config = """
storages:
  adlstore:      
  - reader: ncs_reader
    base_dir: /ncs_data
    assets:
    - name: asset1
      path: path/to/asset1
    - name: asset2
      path: path/to/asset2
  - reader: iroc_reader
    base_dir: /iroc_data 
    assets:
    - name: asset3
      path: path/to/asset3
"""

validation_error_config = """
storages:
  adlstore:      
  - reader: ncs_reader
    base_dir: /ncs_data
    assets:
    - name: asset1
      path: path/to/asset1
wrong_field: 42
"""

duplicate_error_config = """
storages:
  adlstore:      
  - reader: ncs_reader
    base_dir: /ncs_data/first
    assets:
    - name: asset1
      path: to/asset1
  - reader: ncs_reader
    base_dir: /ncs_data/second
    assets:
    - name: asset1
      path: to/asset1
"""


def test_exception_message():
    message = exception_message("Error", "/path/to/config.yaml")
    assert message == "Error. Config path: /path/to/config.yaml"


def test_load_succeeded():
    f = StringIO(succeeded_config)
    config = AssetsConfig.load_from_yaml(f)
    assert config.storages == {
        "adlstore": {
            "asset1": PathSpec(
                reader="ncs_reader", base_dir="/ncs_data", path="path/to/asset1"
            ),
            "asset2": PathSpec(
                reader="ncs_reader", base_dir="/ncs_data", path="path/to/asset2"
            ),
            "asset3": PathSpec(
                reader="iroc_reader", base_dir="/iroc_data", path="path/to/asset3"
            ),
        }
    }
    asset2 = config.get_path("adlstore", "asset2")
    assert asset2 == PathSpec(
        reader="ncs_reader", base_dir="/ncs_data", path="path/to/asset2"
    )


def test_validation_error_exception_message_with_one_error():
    messages = {"field": ["Wrong type.", "Unknown field."]}
    message = validation_error_exception_message(ValidationError(messages))
    assert (
        message
        == 'Validation errors: on field "field" with messages "Wrong type.", "Unknown field."'
    )


def test_validation_error_exception_message_with_two_error():
    messages = {
        "first": ["Wrong type."],
        "second": ["Unknown field."],
    }
    message = validation_error_exception_message(ValidationError(messages))
    assert (
        message
        == 'Validation errors: on field "first" with message "Wrong type."; on field "second" with message "Unknown field."'
    )


def test_validation_error():
    f = StringIO(validation_error_config)
    with pytest.raises(ConfigException):
        AssetsConfig.load_from_yaml(f)


def test_duplicate_error():
    f = StringIO(duplicate_error_config)
    with pytest.raises(ConfigException):
        AssetsConfig.load_from_yaml(f)


def test_load_assets_config():
    assets_config = load_assets_config()
    assert type(assets_config) is AssetsConfig
