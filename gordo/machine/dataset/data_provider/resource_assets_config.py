import pkg_resources

from io import TextIOWrapper
from .assets_config import AssetsConfig


def load_assets_config() -> AssetsConfig:
    # TODO migrate to importlib.resources after deprecating python 3.6 support
    with pkg_resources.resource_stream(
        "gordo.machine.dataset.data_provider.resources", "assets_config.yaml"
    ) as f:
        assets_config = AssetsConfig.load_from_yaml(
            TextIOWrapper(f), "assets_config.yaml"
        )
    return assets_config
