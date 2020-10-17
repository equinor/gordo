import os
from typing import Iterable, IO, Tuple, Optional

import dateutil.parser
import logging

import pytest
import adal

from io import BytesIO

from gordo.machine.dataset.data_provider.providers import DataLakeProvider
from gordo.machine.dataset import dataset
from gordo.machine.dataset.sensor_tag import normalize_sensor_tags
from gordo.machine.dataset.file_system import FileSystem, FileInfo, FileType
from gordo.machine.dataset.data_provider.assets_config import AssetsConfig, PathSpec


class MockFileSystem(FileSystem):
    @property
    def name(self) -> str:
        return "dlstore"

    def open(self, path: str, mode: str = "r") -> IO:
        return BytesIO()

    def exists(self, path: str) -> bool:
        return False

    def isfile(self, path: str) -> bool:
        return False

    def isdir(self, path: str) -> bool:
        return False

    def info(self, path: str) -> FileInfo:
        raise FileNotFoundError(path)

    def ls(
        self, path: str, with_info: bool = True
    ) -> Iterable[Tuple[str, Optional[FileInfo]]]:
        return []

    def walk(
        self, base_path: str, with_info: bool = True
    ) -> Iterable[Tuple[str, Optional[FileInfo]]]:
        return []


@pytest.fixture
def mock_file_system():
    return MockFileSystem()


@pytest.fixture
def mock_assets_config():
    storages = {
        "dlstore": {
            "1755-gra": PathSpec("ncs_reader", "", "path/to/asset1"),
            "1776-troc": PathSpec("ncs_reader", "", "path/to/asset2"),
        }
    }
    return AssetsConfig(storages)


@pytest.fixture
def dataset_config(mock_file_system, mock_assets_config):
    train_start_date = dateutil.parser.isoparse("2017-01-01T08:56:00+00:00")
    train_end_date = dateutil.parser.isoparse("2017-01-01T10:01:00+00:00")
    return {
        "type": "TimeSeriesDataset",
        "train_start_date": train_start_date,
        "train_end_date": train_end_date,
        "tag_list": normalize_sensor_tags(["TRC-FIQ -39-0706", "GRA-EM-23-0003ARV.PV"]),
        "data_provider": DataLakeProvider(
            storage=mock_file_system, assets_config=mock_assets_config
        ),
    }


def test_init(dataset_config):
    config = dataset_config
    dl_backed = dataset._get_dataset(config)
    assert (
        dl_backed is not None
    ), f"Failed to create dataset object of type {config['type']}"


@pytest.mark.skipif(
    os.getenv("INTERACTIVE") is None,
    reason="Skipping test, INTERACTIVE not set in environment variable",
)
def test_get_data_interactive(dataset_config):
    dataset_config = dataset_config
    dataset_config["data_provider"] = DataLakeProvider(interactive=True)
    dl_backed = dataset._get_dataset(dataset_config)
    data = dl_backed.get_data()
    assert len(data) >= 0


@pytest.mark.skipif(
    os.getenv("TEST_SERVICE_AUTH") is None,
    reason="Skipping test, TEST_SERVICE_AUTH not set in environment variable",
)
def test_get_data_serviceauth_in_config(
    mock_file_system, mock_assets_config, default_config
):
    dataset_config = default_config
    dataset_config["data_provider"] = DataLakeProvider(
        storage=mock_file_system,
        assets_config=mock_assets_config,
        dl_service_auth_str=os.getenv("TEST_SERVICE_AUTH"),
    )
    dataset_config["resolution"] = "10T"
    dl_backed = dataset._get_dataset(dataset_config)
    data, _ = dl_backed.get_data()

    assert dataset_config["tag_list"] == list(data.columns.values)

    expected_rows = 7
    assert (
        len(data) == expected_rows
    ), f"Default resolution 10 minutes should give {expected_rows} rows"

    assert (
        not data.isnull().values.any()
    ), "Resulting dataframe should not have any NaNs"
