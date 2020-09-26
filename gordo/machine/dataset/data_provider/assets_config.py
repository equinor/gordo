import yaml

from typing import Optional, Dict, IO
from marshmallow import Schema, fields, ValidationError
from dataclasses import dataclass

from ..exceptions import ConfigException
from ..file_system.base import FileSystem


class AssetItemSchema(Schema):
    name = fields.Str(required=True)
    path = fields.Str(required=True)


class ReaderItemSchema(Schema):
    reader = fields.Str(required=True)
    base_dir = fields.Str(default="")
    assets = fields.List(fields.Nested(AssetItemSchema))


class ConfigSchema(Schema):
    storages = fields.Dict(
        keys=fields.Str,
        values=fields.List(fields.Nested(ReaderItemSchema)),
        required=True,
    )


@dataclass(frozen=True)
class PathSpec:
    reader: str
    base_dir: str
    path: str

    def full_path(self, fs: FileSystem) -> str:
        return fs.join(self.base_dir, self.path)


def exception_message(message: str, file_path: Optional[str] = None) -> str:
    if file_path:
        return message + ". Config path: %s" % file_path
    else:
        return message


def validation_error_exception_message(e: ValidationError) -> str:
    validation_messages = e.normalized_messages()
    messages_list = []
    message_format = 'on field "%s" with message%s %s'
    messages_count = 0
    for field in sorted(validation_messages.keys()):
        messages = validation_messages[field]
        messages_count += len(messages)
        messages_str = ", ".join('"' + msg + '"' for msg in messages)
        message = message_format % (
            field,
            "" if len(messages) <= 1 else "s",
            messages_str,
        )
        messages_list.append(message)
    message = "Validation error%s: %s" % (
        "" if messages_count <= 1 else "s",
        "; ".join(messages_list),
    )
    return message


class AssetsConfig:

    schema = ConfigSchema()

    @classmethod
    def load_from_yaml(
        cls, f: IO[str], file_path: Optional[str] = None
    ) -> "AssetsConfig":
        """
        Loading AssetsConfig from YAML file

        Parameters
        ----------
        f: IO[str]
        file_path
            Source file path. Using only for exception messages

        Returns
        -------

        """
        raw_config = yaml.safe_load(f)
        return cls.load(raw_config, file_path=file_path)

    @classmethod
    def load(cls, raw_config: dict, file_path: Optional[str] = None) -> "AssetsConfig":
        """
        Loading AssetsConfig from a dictionary. See ``load_from_yaml`` method for loading from YAML file

        Examples
        --------
        >>> raw_config = {'storages': {'adlstore': [{'assets': [{'name': 'asset1',
        ...                            'path': 'path/to/asset1'},
        ...                           {'name': 'asset2',
        ...                            'path': 'path/to/asset2'}],
        ...                 'base_dir': '/ncs_data',
        ...                 'reader': 'ncs_reader'}]}}
        >>> config = AssetsConfig.load(raw_config)
        >>> config.get_path("adlstore", "asset2")
        PathSpec(reader='ncs_reader', base_dir='/ncs_data', path='path/to/asset2')

        Parameters
        ----------
        raw_config: dict
            Config source
        file_path
            Source file path. Using only for exception messages


        Returns
        -------
        AssetsConfig

        """
        try:
            valid_config = cls.schema.load(raw_config)
        except ValidationError as e:
            message = validation_error_exception_message(e)
            raise ConfigException(exception_message(message, file_path))
        storages = {}
        for storage, reader_items in valid_config["storages"].items():
            assets: Dict[str, PathSpec] = {}
            for reader_item in reader_items:
                reader = reader_item["reader"]
                base_dir = reader_item["base_dir"]
                for asset in reader_item["assets"]:
                    name = asset["name"]
                    if name in assets:
                        dup = assets[name]
                        message = (
                            f"Found duplicate in storage '{storage}' for asset '{name}', reader '{reader}' "
                            f" and base directory '{base_dir}' with asset from "
                            f"reader '{dup.reader}' and base directory '{dup.base_dir}'"
                        )
                        raise ConfigException(exception_message(message))
                    path_spec = PathSpec(reader, base_dir, asset["path"])
                    assets[name] = path_spec
            storages[storage] = assets
        return cls(storages)

    def __init__(self, storages: Dict[str, Dict[str, PathSpec]]):
        self.storages = storages

    def get_path(self, storage: str, asset: str) -> Optional[PathSpec]:
        storages = self.storages
        if storage not in storages:
            return None
        return storages[storage].get(asset)
