import yaml

from typing import Optional
from marshmallow import Schema, fields, ValidationError

from ..exceptions import ConfigException


class PathItemSchema(Schema):
    asset = fields.Str(required=True)
    path = fields.Str(required=True)


class ConfigSchema(Schema):
    storages = fields.Dict(
        keys=fields.Str,
        values=fields.List(PathItemSchema),
        required=True,
    )


class AssetToPathConfig:

    schema = ConfigSchema()

    @classmethod
    def load_from_yaml(cls, file_path: str):
        with open(file_path, 'r') as f:
            raw_config = yaml.safe_load(f)
        return cls(raw_config, file_path=file_path)

    def __init__(self, raw_config: dict, file_path: Optional[str] = None):
        self.config = self.load_config(raw_config)
        self.file_path = file_path

    def exception_message(self, message: str) -> str:
        if self.file_path:
            return message+". Config path: %s" % self.file_path
        else:
            return message

    def load_config(self, raw_config: dict) -> dict:
        try:
            valid_config = self.schema.load(raw_config)
        except ValidationError as e:
            message = "Validation error: %s" % str(e)
            raise ConfigException(self.exception_message(message))
        storages = {}
        for storage, items_list in valid_config["storages"].items():
            items = {}
            for item in items_list:
                asset, path = item["asset"], item["path"]
                if asset in items:
                    message = "Duplicate asset '%s' for storage '%s'" % (asset, storage)
                    raise ConfigException(self.exception_message(message))
            storages[storage] = items
        return {"storages": storages}

    def get_path(self, storage: str, asset: str):
        storages = self.config["storages"]
        if storage not in storages:
            return None
        return storages[storage].get(asset)
