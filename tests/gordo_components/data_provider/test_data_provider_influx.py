import logging
import dateutil.parser

import pytest

from gordo_components.data_provider.providers import InfluxDataProvider
from gordo_components.client.utils import influx_client_from_uri
from gordo_components.dataset.dataset import _get_dataset

from tests import utils as tu


logger = logging.getLogger(__name__)


def test_read_single_sensor_empty_data_time_range_indexerror(influxdb, caplog):
    """
    Asserts that an IndexError is raised because the dates requested are outside the existing time period
    """
    from_ts = dateutil.parser.isoparse("2017-01-01T09:11:00+00:00")
    to_ts = dateutil.parser.isoparse("2017-01-01T10:30:00+00:00")

    ds = InfluxDataProvider(
        measurement="sensors",
        value_name="Value",
        client=influx_client_from_uri(uri=tu.INFLUXDB_URI, dataframe_client=True),
    )

    with caplog.at_level(logging.CRITICAL):
        with pytest.raises(IndexError):
            ds.read_single_sensor(
                from_ts=from_ts,
                to_ts=to_ts,
                tag=tu.SENSORS_STR_LIST[0],
                measurement="sensors",
            )


def test_read_single_sensor_empty_data_invalid_tag_name_valueerror(influxdb):
    """
    Asserts that a ValueError is raised because the tag name inputted is invalid
    """
    from_ts = dateutil.parser.isoparse("2016-01-01T09:11:00+00:00")
    to_ts = dateutil.parser.isoparse("2016-01-01T10:30:00+00:00")

    ds = InfluxDataProvider(
        measurement="sensors",
        value_name="Value",
        client=influx_client_from_uri(uri=tu.INFLUXDB_URI, dataframe_client=True),
    )
    with pytest.raises(ValueError):
        ds.read_single_sensor(
            from_ts=from_ts,
            to_ts=to_ts,
            tag="tag-does-not-exist",
            measurement="sensors",
        )


def test__list_of_tags_from_influx_validate_tag_names(influxdb):
    """
    Test expected tags in influx match the ones actually in influx.
    """
    ds = InfluxDataProvider(
        measurement="sensors",
        value_name="Value",
        client=influx_client_from_uri(uri=tu.INFLUXDB_URI, dataframe_client=True),
    )
    list_of_tags = ds._list_of_tags_from_influx()
    expected_tags = tu.SENSORS_STR_LIST
    tags = set(list_of_tags)
    assert set(expected_tags) == tags, (
        f"Expected tags = {expected_tags}" f"outputted {tags}"
    )


def test_get_list_of_tags(influxdb):
    ds = InfluxDataProvider(
        measurement="sensors",
        value_name="Value",
        client=influx_client_from_uri(uri=tu.INFLUXDB_URI, dataframe_client=True),
    )
    expected_tags = set(tu.SENSORS_STR_LIST)

    tags = set(ds.get_list_of_tags())
    assert expected_tags == tags

    # The cache does not screw stuff up
    tags = set(ds.get_list_of_tags())
    assert expected_tags == tags


def test_influx_dataset_attrs(influxdb):
    """
    Test expected attributes
    """
    from_ts = dateutil.parser.isoparse("2016-01-01T09:11:00+00:00")
    to_ts = dateutil.parser.isoparse("2016-01-01T10:30:00+00:00")
    tag_list = tu.SENSORTAG_LIST
    config = {
        "type": "TimeSeriesDataset",
        "from_ts": from_ts,
        "to_ts": to_ts,
        "tag_list": tag_list,
    }
    config["data_provider"] = InfluxDataProvider(
        measurement="sensors",
        value_name="Value",
        client=influx_client_from_uri(uri=tu.INFLUXDB_URI, dataframe_client=True),
    )
    dataset = _get_dataset(config)
    assert hasattr(dataset, "get_metadata")

    metadata = dataset.get_metadata()
    assert isinstance(metadata, dict)


def test_influx_load_series_dry_run_raises():
    ds = InfluxDataProvider(measurement="sensors", value_name="Value", client=None)
    from_ts = dateutil.parser.isoparse("2016-01-01T09:11:00+00:00")
    to_ts = dateutil.parser.isoparse("2016-01-01T10:30:00+00:00")
    tag_list = tu.SENSORTAG_LIST
    with pytest.raises(NotImplementedError):
        ds.load_series(from_ts=from_ts, to_ts=to_ts, tag_list=tag_list, dry_run=True)
