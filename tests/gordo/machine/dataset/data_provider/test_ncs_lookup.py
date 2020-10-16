import pytest
import posixpath

from unittest.mock import MagicMock

from gordo.machine.dataset.data_provider.file_type import (
    ParquetFileType,
    CsvFileType,
    TimeSeriesColumns,
)
from gordo.machine.dataset.file_system import FileType, FileInfo
from gordo.machine.dataset.file_system.base import default_join
from gordo.machine.dataset.data_provider.ncs_lookup import (
    NcsLookup,
    TagLocations,
    Location,
)
from gordo.machine.dataset.sensor_tag import SensorTag
from gordo.machine.dataset.data_provider.assets_config import PathSpec

from gordo.machine.dataset.exceptions import ConfigException


@pytest.fixture
def parquet_file_type():
    return ParquetFileType(TimeSeriesColumns("time", "value"))


def test_tag_locations(parquet_file_type):
    tag = SensorTag("tag1", "asset")
    location_2020 = Location("path/2020.parquet", parquet_file_type)
    locations = {
        2020: location_2020,
        2018: Location("path/2018.parquet", parquet_file_type),
    }
    tag_locations = TagLocations(tag, locations)
    assert tag_locations.available()
    assert tag_locations.years() == [2018, 2020]
    assert tag_locations.get_location(2020) is location_2020
    assert tag_locations.get_location(2019) is None
    result = list(tag_locations)
    assert result == [
        (
            SensorTag(name="tag1", asset="asset"),
            2018,
            Location(path="path/2018.parquet", file_type=parquet_file_type),
        ),
        (
            SensorTag(name="tag1", asset="asset"),
            2020,
            Location(path="path/2020.parquet", file_type=parquet_file_type),
        ),
    ]


def test_tag_locations_empty():
    tag = SensorTag("tag1", "asset")
    tag_locations = TagLocations(tag, None)
    assert not tag_locations.available()
    assert len(list(tag_locations)) == 0
    assert len(tag_locations.years()) == 0


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
        ("base/path", FileType.DIRECTORY),
        ("base/path/tag1", FileType.DIRECTORY),
        ("base/path/tag3", FileType.DIRECTORY),
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
    mock.join.side_effect = default_join
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


def reduce_tag_locations(tag_locations_list):
    result = {}
    for tag_locations in tag_locations_list:
        for v in tag_locations:
            tag, year, location = v
            result[(tag.name, year)] = (
                location.path,
                type(location.file_type) if location.file_type is not None else None,
            )
    return result


def test_files_lookup_asgard(default_ncs_lookup: NcsLookup):
    tag = SensorTag("Ásgarðr", "asset")
    years = [2019, 2020]
    result = reduce_tag_locations(
        [default_ncs_lookup.files_lookup("path/%C3%81sgar%C3%B0r", tag, years)]
    )
    assert result == {
        ("Ásgarðr", 2019): (
            "path/%C3%81sgar%C3%B0r/%C3%81sgar%C3%B0r_2019.csv",
            CsvFileType,
        ),
    }


def test_files_lookup_tag2(default_ncs_lookup: NcsLookup):
    tag = SensorTag("tag2", "asset")
    result = reduce_tag_locations(
        [default_ncs_lookup.files_lookup("path/tag2", tag, [2019, 2020])]
    )
    assert result == {
        ("tag2", 2020): ("path/tag2/parquet/tag2_2020.parquet", ParquetFileType),
    }


def test_files_lookup_tag3(default_ncs_lookup: NcsLookup):
    tag = SensorTag("tag3", "asset")
    result = reduce_tag_locations(
        [default_ncs_lookup.files_lookup("path/tag3", tag, [2019, 2020])]
    )
    assert result == {
        ("tag3", 2020): ("path/tag3/parquet/tag3_2020.parquet", ParquetFileType),
    }


@pytest.fixture
def mock_assets_config():
    def get_path_side_effect(storage, asset):
        if asset == "asset":
            return PathSpec("ncs_reader", "", "path")
        elif asset == "asset1":
            return PathSpec("ncs_reader", "", "path1")
        elif asset == "asset5":
            return PathSpec("iroc_reader", "", "path5")
        return None

    mock = MagicMock()
    mock.get_path.side_effect = get_path_side_effect
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
        (SensorTag(name="tag1", asset="asset"), None),
        (SensorTag(name="tag4", asset="asset"), None),
        (SensorTag(name="tag5", asset="asset1"), "path1/tag5"),
    ]


def test_assets_config_tags_lookup_base_dir(
    default_ncs_lookup: NcsLookup, mock_assets_config
):
    tags = [
        SensorTag("tag1", "asset"),
        SensorTag("tag2", "asset"),
        SensorTag("tag3", "asset1"),
    ]
    result = {}
    for tag, path in default_ncs_lookup.assets_config_tags_lookup(
        mock_assets_config, tags, base_dir="base/path"
    ):
        result[tag] = path
    assert result == {
        SensorTag(name="tag1", asset="asset"): "base/path/tag1",
        SensorTag(name="tag3", asset="asset1"): "base/path/tag3",
        SensorTag(name="tag2", asset="asset"): None,
    }


def test_assets_config_tags_lookup_exceptions(
    default_ncs_lookup: NcsLookup, mock_assets_config
):
    tags = [
        SensorTag("Ásgarðr", "asset"),
        SensorTag("tag10", ""),
    ]
    with pytest.raises(ValueError):
        list(default_ncs_lookup.assets_config_tags_lookup(mock_assets_config, tags))
    tags = [
        SensorTag("Ásgarðr", "asset"),
        SensorTag("tag10", "asset10"),
    ]
    with pytest.raises(ValueError):
        list(default_ncs_lookup.assets_config_tags_lookup(mock_assets_config, tags))


@pytest.mark.parametrize(
    "threads_count", [1, 2, 10],
)
def test_lookup_default(
    default_ncs_lookup: NcsLookup, mock_assets_config, threads_count
):
    tags = [
        SensorTag("Ásgarðr", "asset"),
        SensorTag("tag1", "asset"),
        SensorTag("tag2", "asset"),
        SensorTag("tag4", "asset"),
        SensorTag("tag5", "asset1"),
    ]
    result = list(
        default_ncs_lookup.lookup(
            mock_assets_config, tags, [2019, 2020], threads_count=threads_count
        )
    )
    assert reduce_tag_locations(result) == {
        ("Ásgarðr", 2019): (
            "path/%C3%81sgar%C3%B0r/%C3%81sgar%C3%B0r_2019.csv",
            CsvFileType,
        ),
        ("tag2", 2020): ("path/tag2/parquet/tag2_2020.parquet", ParquetFileType),
        ("tag5", 2020): ("path1/tag5/parquet/tag5_2020.parquet", ParquetFileType),
    }


@pytest.mark.parametrize(
    "threads_count", [None, 0],
)
def test_lookup_exceptions(
    default_ncs_lookup: NcsLookup, mock_assets_config, threads_count
):
    tags = [
        SensorTag("Ásgarðr", "asset"),
        SensorTag("tag1", "asset"),
    ]
    with pytest.raises(ConfigException):
        list(
            default_ncs_lookup.lookup(
                mock_assets_config, tags, [2019, 2020], threads_count=threads_count
            )
        )


def test_assets_config_wrong_reader(default_ncs_lookup: NcsLookup, mock_assets_config):
    tags = [
        SensorTag("tag4", "asset5"),
    ]
    with pytest.raises(ValueError):
        list(default_ncs_lookup.assets_config_tags_lookup(mock_assets_config, tags))
