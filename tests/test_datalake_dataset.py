import os
from unittest import TestCase

import dateutil.parser
import unittest
import adal

from gordo_components.dataset import dataset
from gordo_components.data_provider.providers import DataLakeBackedDataset


class DataLakeTestCase(unittest.TestCase):
    @classmethod
    def setUp(self):
        self.datalake_config = {"storename": "dataplatformdlsprod"}
        self.tag_list = ["TRC-FIQ -39-0706", "GRA-EM  -23-0003ARV.PV"]

        from_ts = dateutil.parser.isoparse("2017-01-01T08:56:00+00:00")
        to_ts = dateutil.parser.isoparse("2017-01-01T10:01:00+00:00")

        self.dataset_config = {
            "type": "DataLakeBackedDataset",
            "from_ts": from_ts,
            "to_ts": to_ts,
            "datalake_config": self.datalake_config,
            "tag_list": self.tag_list,
        }

    def test_init(self):
        dl_backed = dataset._get_dataset(self.dataset_config)
        self.assertIsNotNone(
            dl_backed,
            f"Failed to create dataset object of type {self.dataset_config['type']}",
        )

    def test_get_data_serviceauth_fail(self):
        self.datalake_config["dl_service_auth_str"] = "TENTANT_UNKNOWN:BOGUS:PASSWORD"
        dl_backed = dataset._get_dataset(self.dataset_config)
        self.assertRaises(adal.adal_error.AdalError, dl_backed.get_data)

    def test_get_metadata(self):
        dl_backed = dataset._get_dataset(self.dataset_config)
        metadata = dl_backed.get_metadata()
        self.assertEqual(metadata["train_start_date"], self.dataset_config["from_ts"])
        self.assertEqual(metadata["train_end_date"], self.dataset_config["to_ts"])
        self.assertEqual(metadata["tag_list"], self.dataset_config["tag_list"])
        self.assertEqual(metadata["resolution"], "10T")

        self.dataset_config["resolution"] = "10M"
        dl_backed = dataset._get_dataset(self.dataset_config)
        metadata = dl_backed.get_metadata()
        self.assertEqual(metadata["resolution"], self.dataset_config["resolution"])

    @unittest.skipIf(
        os.getenv("INTERACTIVE") is None,
        "Skipping test, INTERACTIVE not set in environment variable",
    )
    def test_get_data_interactive(self):
        self.datalake_config["interactive"] = True
        dl_backed = dataset._get_dataset(self.dataset_config)
        data = dl_backed.get_data()
        self.assertGreaterEqual(len(data), 0)

    @unittest.skipIf(
        os.getenv("TEST_SERVICE_AUTH") is None,
        "Skipping test, TEST_SERVICE_AUTH not set in environment variable",
    )
    def test_get_data_serviceauth_in_config(self):
        self.datalake_config["dl_service_auth_str"] = os.getenv("TEST_SERVICE_AUTH")
        self.dataset_config["resolution"] = "10T"
        dl_backed = dataset._get_dataset(self.dataset_config)
        data, _ = dl_backed.get_data()

        self.assertListEqual(self.tag_list, list(data.columns.values))

        expected_rows = 7
        self.assertEqual(
            len(data),
            expected_rows,
            f"Default resolution 10 minutes should give {expected_rows} rows",
        )

        self.assertFalse(
            data.isnull().values.any(), "Resulting dataframe should not have any NaNs"
        )

    def test_get_datalake_token_wrong_args(self):
        with self.assertRaises(ValueError):
            DataLakeBackedDataset.get_datalake_token(interactive=False)
