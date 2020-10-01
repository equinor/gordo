import pytest
import posixpath

from unittest.mock import MagicMock

from gordo.machine.dataset.data_provider.file_type import ParquetFileType, CsvFileType
from gordo.machine.dataset.file_system import FileType, FileInfo
from gordo.machine.dataset.data_provider.ncs_lookup import NcsLookup, TagLocation
from gordo.machine.dataset.sensor_tag import SensorTag
from gordo.machine.dataset.data_provider.assets_config import PathSpec

from typing import List, Tuple, Dict, Optional


@pytest.fixture
def dir_tree():
    return [
        # tag.name = Ásgarðr
        ("path/%C3%81sgar%C3%B0r", FileType.DIRECTORY),
        ("path/%C3%81sgar%C3%B0r/%C3%81sgar%C3%B0r_2019.csv", FileType.FILE),
        ("path/tag2", FileType.DIRECTORY),
        ("path/tag2/parquet", FileType.DIRECTORY),
        ("path/tag2/parquet/tag2_2020.parquet", FileType.FILE),
        ("path/tag3", FileType.DIRECTORY),
        ("path/tag3/parquet", FileType.DIRECTORY),
        ("path/tag3/parquet/tag3_2020.parquet", FileType.FILE),
        ("path/tag3/tag3_2020.csv", FileType.FILE),
        ("path1/tag5", FileType.DIRECTORY),
        ("path1/tag5/parquet", FileType.DIRECTORY),
        ("path1/tag5/parquet/tag5_2020.parquet", FileType.FILE),
    ]


@pytest.fixture
def mock_file_system(dir_tree):
    def ls_side_effect(path, with_info=True):
        for file_path, file_type in dir_tree:
            dir_path, _ = posixpath.split(file_path)
            if dir_path == path:
                yield file_path, FileInfo(file_type, 0) if with_info else None

    def walk_side_effect(base_path, with_info=True):
        for file_path, file_type in dir_tree:
            if file_path.find(base_path) == 0:
                yield file_path, FileInfo(file_type, 0) if with_info else None

    def exists_side_effect(path):
        for file_path, _ in dir_tree:
            if path == file_path:
                return True
        return False

    mock = MagicMock()
    mock.exists.side_effect = exists_side_effect
    mock.ls.side_effect = ls_side_effect
    mock.walk.side_effect = walk_side_effect
    mock.join.side_effect = posixpath.join
    mock.split.side_effect = posixpath.split
    return mock


def test_mock_file_system(mock_file_system):
    result = list(mock_file_system.ls("path"))
    assert result == [
        ("path/%C3%81sgar%C3%B0r", FileInfo(file_type=FileType.DIRECTORY, size=0),),
        ("path/tag2", FileInfo(file_type=FileType.DIRECTORY, size=0),),
        ("path/tag3", FileInfo(file_type=FileType.DIRECTORY, size=0),),
    ]
    result = list(mock_file_system.walk("path/tag2"))
    assert result == [
        ("path/tag2", FileInfo(file_type=FileType.DIRECTORY, size=0),),
        ("path/tag2/parquet", FileInfo(file_type=FileType.DIRECTORY, size=0),),
        (
            "path/tag2/parquet/tag2_2020.parquet",
            FileInfo(file_type=FileType.FILE, size=0),
        ),
    ]
    assert mock_file_system.exists("path/%C3%81sgar%C3%B0r")
    assert not mock_file_system.exists("path/out")
    result = list(mock_file_system.ls("path1"))
    assert result == [
        (
            "path1/tag5",
            FileInfo(
                file_type=FileType.DIRECTORY,
                size=0,
                access_time=None,
                modify_time=None,
                create_time=None,
            ),
        )
    ]


@pytest.fixture
def default_ncs_lookup(mock_file_system):
    return NcsLookup.create(mock_file_system)


def test_tag_dirs_lookup(default_ncs_lookup: NcsLookup):
    tags = [
        SensorTag("Ásgarðr", "asset"),
        SensorTag("tag1", "asset"),
        SensorTag("tag2", "asset"),
        SensorTag("tag4", "asset"),
    ]
    result = {}
    for tag, path in default_ncs_lookup.tag_dirs_lookup("path", tags):
        result[tag.name] = path
    assert result == {
        "Ásgarðr": "path/%C3%81sgar%C3%B0r",
        "tag2": "path/tag2",
        "tag1": None,
        "tag4": None,
    }


def reduce_tag_locations(tag_locations):
    result = {}
    for location in tag_locations:
        result[(location.tag.name, location.year)] = (
            location.path,
            type(location.file_type) if location.file_type is not None else None,
        )
    return result


def test_files_lookup_asgard(default_ncs_lookup: NcsLookup):
    tag = SensorTag("Ásgarðr", "asset")
    result = []
    for location in default_ncs_lookup.files_lookup(
        "path/%C3%81sgar%C3%B0r", tag, [2019, 2020]
    ):
        result.append(location)
    assert len(result) == 2
    assert reduce_tag_locations(result) == {
        ("Ásgarðr", 2019): (
            "path/%C3%81sgar%C3%B0r/%C3%81sgar%C3%B0r_2019.csv",
            CsvFileType,
        ),
        ("Ásgarðr", 2020): (None, None),
    }


def test_files_lookup_tag2(default_ncs_lookup: NcsLookup):
    tag = SensorTag("tag2", "asset")
    result = []
    for location in default_ncs_lookup.files_lookup("path/tag2", tag, [2019, 2020]):
        result.append(location)
    assert len(result) == 2
    assert reduce_tag_locations(result) == {
        ("tag2", 2020): ("path/tag2/parquet/tag2_2020.parquet", ParquetFileType),
        ("tag2", 2019): (None, None),
    }


def test_files_lookup_tag3(default_ncs_lookup: NcsLookup):
    tag = SensorTag("tag3", "asset")
    result = []
    for location in default_ncs_lookup.files_lookup("path/tag3", tag, [2019, 2020]):
        result.append(location)
    assert len(result) == 2
    assert reduce_tag_locations(result) == {
        ("tag3", 2020): ("path/tag3/parquet/tag3_2020.parquet", ParquetFileType),
        ("tag3", 2019): (None, None),
    }


@pytest.fixture
def mock_assets_config():
    def get_paths_side_effect(storage, asset):
        if asset == "asset":
            return PathSpec("ncs_reader", "", "path")
        elif asset == "asset1":
            return PathSpec("ncs_reader", "", "path1")
        return None

    mock = MagicMock()
    mock.get_path.side_effect = get_paths_side_effect
    return mock


def test_assets_config_tags_lookup(default_ncs_lookup: NcsLookup, mock_assets_config):
    tags = [
        SensorTag("Ásgarðr", "asset"),
        SensorTag("tag1", "asset"),
        SensorTag("tag2", "asset"),
        SensorTag("tag4", "asset"),
        SensorTag("tag5", "asset1"),
    ]
    result = list(
        default_ncs_lookup.assets_config_tags_lookup(mock_assets_config, tags)
    )
    assert result == [
        (SensorTag(name="Ásgarðr", asset="asset"), "path/%C3%81sgar%C3%B0r"),
        (SensorTag(name="tag2", asset="asset"), "path/tag2"),
        (SensorTag(name='tag1', asset='asset'), None),
        (SensorTag(name='tag4', asset='asset'), None),
        (SensorTag(name="tag5", asset="asset1"), "path1/tag5"),
    ]
