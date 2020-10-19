import os

from unittest.mock import patch, Mock
import pytest

import dateutil.parser

from gordo.machine.dataset.data_provider.ncs_reader import NcsReader
from gordo.machine.dataset.data_provider.assets_config import AssetsConfig, PathSpec
from gordo.machine.dataset.sensor_tag import normalize_sensor_tags
from gordo.machine.dataset.sensor_tag import SensorTag
from gordo.machine.dataset.file_system.adl1 import ADLGen1FileSystem


class AzureDLFileSystemMock:
    def exists(self, file_path):
        return os.path.exists(file_path)

    def info(self, file_path):
        info = {"length": os.path.getsize(file_path)}
        if os.path.isfile(file_path):
            info["type"] = "FILE"
        elif os.path.isdir(file_path):
            info["type"] = "DIRECTORY"
        return info

    def open(self, file_path, mode):
        return open(file_path, mode)

    def ls(self, dir_path, detail):
        result = []
        for file_name in os.listdir(dir_path):
            full_path = os.path.join(dir_path, file_name)
            if detail:
                info = self.info(full_path)
                info["name"] = full_path
                result.append(info)
            else:
                result.append(full_path)
        return result


@pytest.fixture
def assets_config():
    base_dir = os.path.dirname(os.path.realpath(__file__))
    assets_paths = [
        ("gordoplatform", os.path.join("data", "datalake", "gordoplatform")),
        ("1776-troc", os.path.join("data", "datalake")),
    ]
    adl1_assets = {
        asset: PathSpec("ncs_reader", base_dir, path) for asset, path in assets_paths
    }
    storages = {"adl1": adl1_assets}
    return AssetsConfig(storages)


@pytest.fixture
def ncs_reader(assets_config):
    return NcsReader(
        ADLGen1FileSystem(AzureDLFileSystemMock(), "adl1"), assets_config=assets_config
    )


@pytest.fixture
def dates():
    return (
        dateutil.parser.isoparse("2000-01-01T08:56:00+00:00"),
        dateutil.parser.isoparse("2001-09-01T10:01:00+00:00"),
    )


@pytest.mark.parametrize(
    "tag_to_check",
    [normalize_sensor_tags(["TRC-123"])[0], SensorTag("XYZ-123", "1776-TROC")],
)
def test_can_handle_tag_ok(tag_to_check, ncs_reader):
    assert ncs_reader.can_handle_tag(tag_to_check)


@pytest.mark.parametrize(
    "tag_to_check", [SensorTag("TRC-123", None), SensorTag("XYZ-123", "123-XXX")]
)
def test_can_handle_tag_notok(tag_to_check, ncs_reader):
    assert not ncs_reader.can_handle_tag(tag_to_check)


def test_can_handle_tag_unknow_prefix_raise(ncs_reader):
    with pytest.raises(ValueError):
        ncs_reader.can_handle_tag(normalize_sensor_tags(["XYZ-123"])[0])


def test_can_handle_tag_non_supported_asset_with_base_path(ncs_reader, assets_config):
    tag = SensorTag("WEIRD-123", "UNKNOWN-ASSET")
    assert not ncs_reader.can_handle_tag(tag)

    ncs_reader_with_base = NcsReader(
        ADLGen1FileSystem(AzureDLFileSystemMock(), "adl1"),
        assets_config=assets_config,
        dl_base_path="/this/is/a/base/path",
    )
    assert ncs_reader_with_base.can_handle_tag(tag)


def test_load_series_need_base_path(ncs_reader, dates, assets_config):
    tag = SensorTag("WEIRD-123", "BASE-PATH-ASSET")
    with pytest.raises(ValueError):
        for _ in ncs_reader.load_series(dates[0], dates[1], [tag]):
            pass

    path_to_weird_base_path_asset = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "data",
        "datalake",
        "base_path_asset",
    )
    ncs_reader_with_base = NcsReader(
        ADLGen1FileSystem(AzureDLFileSystemMock(), "adl1"),
        assets_config=assets_config,
        dl_base_path=path_to_weird_base_path_asset,
    )
    for tag_series in ncs_reader_with_base.load_series(dates[0], dates[1], [tag]):
        assert len(tag_series) == 20


@pytest.mark.parametrize(
    "bad_tag_lists", [["TRC-123"], [{"name": "TRC-123", "asset": None}]]
)
def test_load_series_tag_as_string_fails(bad_tag_lists, dates, ncs_reader):
    with pytest.raises(AttributeError):
        for _ in ncs_reader.load_series(dates[0], dates[1], bad_tag_lists):
            pass


def test_load_series_need_asset_hint(dates, ncs_reader):
    with pytest.raises(ValueError):
        for _ in ncs_reader.load_series(
            dates[0], dates[1], [SensorTag("XYZ-123", None)]
        ):
            pass

    valid_tag_list_with_asset = [SensorTag("XYZ-123", "gordoplatform")]
    for frame in ncs_reader.load_series(dates[0], dates[1], valid_tag_list_with_asset):
        assert len(frame) == 20


def test_load_series_known_prefix(dates, ncs_reader):
    valid_tag_list_no_asset = normalize_sensor_tags(["TRC-123", "TRC-321"])
    for frame in ncs_reader.load_series(dates[0], dates[1], valid_tag_list_no_asset):
        assert len(frame) == 20


@pytest.mark.parametrize(
    "start_date, end_date, frame_len",
    [
        [
            dateutil.parser.isoparse("2000-01-01T08:56:00+00:00"),
            dateutil.parser.isoparse("2021-09-01T10:01:00+00:00"),
            20,
        ],
        [
            # Test something not valid (where there could not be any data)
            dateutil.parser.isoparse("2025-01-01T00:00:00+00:00"),
            dateutil.parser.isoparse("2030-01-01T00:00:00+00:00"),
            0,
        ],
    ],
)
def test_load_series_invalid_year(start_date, end_date, frame_len, ncs_reader):
    valid_tag_list = normalize_sensor_tags(["TRC-123"])
    frame = next(ncs_reader.load_series(start_date, end_date, valid_tag_list))
    assert len(frame) == frame_len


def test_ncs_reader_valid_tag_path():
    with pytest.raises(FileNotFoundError):
        NcsReader._verify_tag_path_exist(
            ADLGen1FileSystem(AzureDLFileSystemMock(), "adl1"), "not/valid/path"
        )


def test_load_series_dry_run(dates, ncs_reader):
    valid_tag_list_no_asset = normalize_sensor_tags(["TRC-123", "TRC-321"])
    for frame in ncs_reader.load_series(
        dates[0], dates[1], valid_tag_list_no_asset, dry_run=True
    ):
        assert len(frame) == 0


@pytest.mark.parametrize("remove_status_codes", [[], [0]])
def test_load_series_with_filter_bad_data(dates, remove_status_codes, assets_config):

    ncs_reader = NcsReader(
        ADLGen1FileSystem(AzureDLFileSystemMock(), "adl1"),
        assets_config=assets_config,
        remove_status_codes=remove_status_codes,
    )

    valid_tag_list = normalize_sensor_tags(["TRC-322"])
    series_gen = ncs_reader.load_series(dates[0], dates[1], valid_tag_list)
    # Checks if the bad data from the files under tests/gordo/data_provider/data/datalake/TRC-322
    # are filtered out. 20 rows exists, 5 of then have the value 0.

    n_expected = 15 if remove_status_codes != [] else 20
    assert all(len(series) == n_expected for series in series_gen)


def test_parquet_files_lookup(dates, assets_config):
    ncs_reader = NcsReader(
        ADLGen1FileSystem(AzureDLFileSystemMock(), "adl1"),
        assets_config=assets_config,
        remove_status_codes=[0],
    )

    valid_tag_list = normalize_sensor_tags(["TRC-323"])
    series_gen = ncs_reader.load_series(dates[0], dates[1], valid_tag_list)
    tags_series = [v for v in series_gen]
    assert len(tags_series) == 1
    trc_323_series = tags_series[0]
    assert trc_323_series.name == "TRC-323"
    assert trc_323_series.dtype.name == "float64"
    assert len(trc_323_series) == 20


def test_with_conflicted_file_types(dates, assets_config):
    ncs_reader = NcsReader(
        ADLGen1FileSystem(AzureDLFileSystemMock(), "adl1"),
        assets_config=assets_config,
        remove_status_codes=[0],
    )

    valid_tag_list = normalize_sensor_tags(["TRC-324"])
    series_gen = ncs_reader.load_series(dates[0], dates[1], valid_tag_list)
    tags_series = [v for v in series_gen]
    assert len(tags_series) == 1
    trc_324_series = tags_series[0]
    # Parquet file should be with 15 rows
    assert len(trc_324_series) == 15


def test_with_conflicted_file_types_with_preferable_csv(dates, assets_config):
    ncs_reader = NcsReader(
        ADLGen1FileSystem(AzureDLFileSystemMock(), "adl1"),
        assets_config=assets_config,
        remove_status_codes=[0],
        lookup_for=["csv"],
    )

    valid_tag_list = normalize_sensor_tags(["TRC-324"])
    series_gen = ncs_reader.load_series(dates[0], dates[1], valid_tag_list)
    tags_series = [v for v in series_gen]
    assert len(tags_series) == 1
    trc_324_series = tags_series[0]
    # CSV file should be with 1 row
    assert len(trc_324_series) == 1
