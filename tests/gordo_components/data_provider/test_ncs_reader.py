import os

from unittest.mock import patch
import pytest

import dateutil.parser

from gordo_components.data_provider.ncs_reader import NcsReader
from gordo_components.dataset.sensor_tag import normalize_sensor_tags
from gordo_components.dataset.sensor_tag import SensorTag


class AzureDLFileSystemMock:
    def info(self, file_path):
        return {"length": os.path.getsize(file_path)}

    def open(self, file_path, mode):
        return open(file_path, mode)


@pytest.fixture
def ncs_reader():
    return NcsReader(AzureDLFileSystemMock())


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

    path_to_xyz = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "data", "datalake", "gordoplatform"
    )
    with patch(
        "gordo_components.data_provider.ncs_reader.NcsReader.ASSET_TO_PATH",
        {"gordoplatform": path_to_xyz},
    ):
        valid_tag_list_with_asset = [SensorTag("XYZ-123", "gordoplatform")]
        for frame in ncs_reader.load_series(
            dates[0], dates[1], valid_tag_list_with_asset
        ):
            assert len(frame) == 20


@patch(
    "gordo_components.data_provider.ncs_reader.NcsReader.ASSET_TO_PATH",
    {
        "1776-troc": os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "data", "datalake"
        )
    },
)
def test_load_series_known_prefix(dates, ncs_reader):
    valid_tag_list_no_asset = normalize_sensor_tags(["TRC-123", "TRC-321"])
    for frame in ncs_reader.load_series(dates[0], dates[1], valid_tag_list_no_asset):
        assert len(frame) == 20


@patch(
    "gordo_components.data_provider.ncs_reader.NcsReader.ASSET_TO_PATH",
    {
        "1776-troc": os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "data", "datalake"
        )
    },
)
def test_load_series_dry_run(dates, ncs_reader):
    valid_tag_list_no_asset = normalize_sensor_tags(["TRC-123", "TRC-321"])
    for frame in ncs_reader.load_series(
        dates[0], dates[1], valid_tag_list_no_asset, dry_run=True
    ):
        assert len(frame) == 0
