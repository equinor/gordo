import os
import dateutil.parser
import logging

import pytest
import adal

from gordo.machine.dataset.data_provider.providers import DataLakeProvider
from gordo.machine.dataset import dataset
from gordo.machine.dataset.sensor_tag import normalize_sensor_tags


def _get_default_dataset_config():
    train_start_date = dateutil.parser.isoparse("2017-01-01T08:56:00+00:00")
    train_end_date = dateutil.parser.isoparse("2017-01-01T10:01:00+00:00")
    return {
        "type": "TimeSeriesDataset",
        "train_start_date": train_start_date,
        "train_end_date": train_end_date,
        "tag_list": normalize_sensor_tags(["TRC-FIQ -39-0706", "GRA-EM-23-0003ARV.PV"]),
        "data_provider": DataLakeProvider(),
    }


def test_get_data_serviceauth_fail(caplog):
    train_start_date = dateutil.parser.isoparse("2017-01-01T08:56:00+00:00")
    train_end_date = dateutil.parser.isoparse("2017-01-01T10:01:00+00:00")

    dataset_config = _get_default_dataset_config()
    dataset_config["train_start_date"] = train_start_date
    dataset_config["train_end_date"] = train_end_date
    dataset_config["data_provider"] = DataLakeProvider(
        dl_service_auth_str="TENTANT_UNKNOWN:BOGUS:PASSWORD"
    )

    dl_backed = dataset._get_dataset(dataset_config)

    with pytest.raises(adal.adal_error.AdalError), caplog.at_level(logging.CRITICAL):
        dl_backed.get_data()


def test_init():
    config = _get_default_dataset_config()
    dl_backed = dataset._get_dataset(config)
    assert (
        dl_backed is not None
    ), f"Failed to create dataset object of type {config['type']}"


@pytest.mark.skipif(
    os.getenv("INTERACTIVE") is None,
    reason="Skipping test, INTERACTIVE not set in environment variable",
)
def test_get_data_interactive():
    dataset_config = _get_default_dataset_config()
    dataset_config["data_provider"] = DataLakeProvider(interactive=True)
    dl_backed = dataset._get_dataset(dataset_config)
    data = dl_backed.get_data()
    assert len(data) >= 0


@pytest.mark.skipif(
    os.getenv("TEST_SERVICE_AUTH") is None,
    reason="Skipping test, TEST_SERVICE_AUTH not set in environment variable",
)
def test_get_data_serviceauth_in_config():
    dataset_config = _get_default_dataset_config()
    dataset_config["data_provider"] = DataLakeProvider(
        dl_service_auth_str=os.getenv("TEST_SERVICE_AUTH")
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
