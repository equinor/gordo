import unittest
from unittest.mock import patch
import os
import dateutil

from gordo_components.data_provider.ncs_reader import NcsReader
from gordo_components.dataset.sensor_tag import normalize_sensor_tag
from gordo_components.dataset.sensor_tag import SensorTag


class AzureDLFileSystemMock:
    def info(self, file_path):
        return {"length": os.path.getsize(file_path)}

    def open(self, file_path, mode):
        return open(file_path, mode)


class NcsReaderTestCase(unittest.TestCase):
    datapath = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "data", "datalake"
    )

    @classmethod
    def setUp(self):
        azure_client_mock = AzureDLFileSystemMock()
        self.ncs_reader = NcsReader(azure_client_mock)
        self.from_ts = dateutil.parser.isoparse("2000-01-01T08:56:00+00:00")
        self.to_ts = dateutil.parser.isoparse("2001-09-01T10:01:00+00:00")

    def test_can_handle_tag(self):
        self.assertFalse(self.ncs_reader.can_handle_tag(SensorTag("TRC-123", None)))

        self.assertTrue(self.ncs_reader.can_handle_tag(normalize_sensor_tag("TRC-123")))

        with self.assertRaises(ValueError):
            self.ncs_reader.can_handle_tag(normalize_sensor_tag("XYZ-123"))

        self.assertTrue(
            self.ncs_reader.can_handle_tag(SensorTag("XYZ-123", "1776-TROC"))
        )
        self.assertFalse(
            self.ncs_reader.can_handle_tag(SensorTag("XYZ-123", "123-XXX"))
        )

    def test_load_series_tag_as_string_fails(self):
        with self.assertRaises(AttributeError):
            for _ in self.ncs_reader.load_series(self.from_ts, self.to_ts, ["TRC-123"]):
                pass

    def test_load_series_tag_as_dict_fails(self):
        with self.assertRaises(AttributeError):
            for _ in self.ncs_reader.load_series(
                self.from_ts, self.to_ts, [{"name": "TRC-123", "asset": None}]
            ):
                pass

    def test_load_series_need_asset_hint(self):
        with self.assertRaises(ValueError):
            for frame in self.ncs_reader.load_series(
                self.from_ts, self.to_ts, [SensorTag("XYZ-123", None)]
            ):
                pass

        path_to_xyz = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "data",
            "datalake",
            "gordoplatform",
        )
        with patch(
            "gordo_components.data_provider.ncs_reader.NcsReader.ASSET_TO_PATH",
            {"gordoplatform": path_to_xyz},
        ):
            valid_tag_list_with_asset = [SensorTag("XYZ-123", "gordoplatform")]
            for frame in self.ncs_reader.load_series(
                self.from_ts, self.to_ts, valid_tag_list_with_asset
            ):
                self.assertEqual(len(frame), 20)

    @patch(
        "gordo_components.data_provider.ncs_reader.NcsReader.ASSET_TO_PATH",
        {"1776-troc": datapath},
    )
    def test_load_series_known_prefix(self):
        valid_tag_list_no_asset = [
            normalize_sensor_tag("TRC-123"),
            normalize_sensor_tag("TRC-321"),
        ]
        for frame in self.ncs_reader.load_series(
            self.from_ts, self.to_ts, valid_tag_list_no_asset
        ):
            self.assertEqual(len(frame), 20)
