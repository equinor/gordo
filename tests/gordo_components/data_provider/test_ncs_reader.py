import os

from unittest.mock import patch
import pytest

import dateutil.parser

from gordo_components.data_provider.ncs_reader import NcsReader
from gordo_components.data_provider.providers import DataLakeProvider
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


def test_can_handle_tag_non_supported_asset_with_base_path(ncs_reader):
    tag = SensorTag("WEIRD-123", "UNKNOWN-ASSET")
    assert not ncs_reader.can_handle_tag(tag)

    ncs_reader_with_base = NcsReader(
        AzureDLFileSystemMock(), dl_base_path="/this/is/a/base/path"
    )
    assert ncs_reader_with_base.can_handle_tag(tag)


def test_load_series_need_base_path(ncs_reader, dates):
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
        AzureDLFileSystemMock(), dl_base_path=path_to_weird_base_path_asset
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


@patch(
    "gordo_components.data_provider.ncs_reader.NcsReader.ASSET_TO_PATH",
    {
        "1776-troc": os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "data", "datalake"
        )
    },
)
@pytest.mark.parametrize("remove_status_codes", [[], [0]])
def test_load_series_with_filter_bad_data(dates, remove_status_codes):

    ncs_reader = NcsReader(
        AzureDLFileSystemMock(), remove_status_codes=remove_status_codes
    )

    valid_tag_list = normalize_sensor_tags(["TRC-322"])
    series_gen = ncs_reader.load_series(dates[0], dates[1], valid_tag_list)
    # Checks if the bad data from the files under tests/gordo_components/data_provider/data/datalake/TRC-322
    # are filtered out. 20 rows exists, 5 of then have the value 0.

    n_expected = 15 if remove_status_codes != [] else 20
    assert all(len(series) == n_expected for series in series_gen)


@pytest.mark.parametrize("remove_status_codes", [[], [0]])
def test_ncs_reader_kwargs_contains_remove_status_codes(remove_status_codes):
    # Creates a DataLakeProvider with remove_status_codes as kwargs
    data_provider = DataLakeProvider(
        interactive=False, remove_status_codes=remove_status_codes
    )

    # Set the data_provider's client to the AzureDLFileSystemMock as interactive can be False.
    data_provider.client = AzureDLFileSystemMock()
    # Get the ncs_reader from data_provider.
    ncs_reader = data_provider._get_sub_dataproviders()[0]

    # Cheks that the kwargs remove_status_codes has been passed to the sub_provider
    expected = [] if remove_status_codes == [] else [0]
    assert ncs_reader.remove_status_codes == expected
