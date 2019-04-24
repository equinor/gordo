import logging
import dateutil.parser

import pytest

from gordo_components.data_provider.providers import InfluxDataProvider
from gordo_components.client.utils import influx_client_from_uri
from gordo_components.dataset import _get_dataset

from tests.gordo_components.server.test_gordo_server import SENSORS


logger = logging.getLogger(__name__)


INFLUXDB_URI = "root:root@localhost:8086/testdb"


@pytest.mark.parametrize(
    "influxdb", [(SENSORS, "testdb", "root", "root", "sensors")], indirect=True
)
def test_read_single_sensor_empty_data_time_range_indexerror(influxdb, caplog):
    """
    Asserts that an IndexError is raised because the dates requested are outside the existing time period
    """
    from_ts = dateutil.parser.isoparse("2017-01-01T09:11:00+00:00")
    to_ts = dateutil.parser.isoparse("2017-01-01T10:30:00+00:00")

    ds = InfluxDataProvider(
        measurement="sensors",
        value_name="Value",
        client=influx_client_from_uri(uri=INFLUXDB_URI, dataframe_client=True),
    )

    with caplog.at_level(logging.CRITICAL):
        with pytest.raises(IndexError):
            ds.read_single_sensor(
                from_ts=from_ts, to_ts=to_ts, tag=SENSORS[0], measurement="sensors"
            )


@pytest.mark.parametrize(
    "influxdb", [(SENSORS, "testdb", "root", "root", "sensors")], indirect=True
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
        client=influx_client_from_uri(uri=INFLUXDB_URI, dataframe_client=True),
    )
    with pytest.raises(ValueError):
        ds.read_single_sensor(
            from_ts=from_ts,
            to_ts=to_ts,
            tag="tag-does-not-exist",
            measurement="sensors",
        )


@pytest.mark.parametrize(
    "influxdb", [(SENSORS, "testdb", "root", "root", "sensors")], indirect=True
)
def test__list_of_tags_from_influx_validate_tag_names(influxdb):
    """
    Test expected tags in influx match the ones actually in influx.
    """
    ds = InfluxDataProvider(
        measurement="sensors",
        value_name="Value",
        client=influx_client_from_uri(uri=INFLUXDB_URI, dataframe_client=True),
    )
    list_of_tags = ds._list_of_tags_from_influx()
    expected_tags = SENSORS
    tags = set(list_of_tags)
    assert set(expected_tags) == tags, f"Expected tags = {SENSORS}" f"outputted {tags}"


@pytest.mark.parametrize(
    "influxdb", [(SENSORS, "testdb", "root", "root", "sensors")], indirect=True
)
def test_get_list_of_tags(influxdb):
    ds = InfluxDataProvider(
        measurement="sensors",
        value_name="Value",
        client=influx_client_from_uri(uri=INFLUXDB_URI, dataframe_client=True),
    )
    expected_tags = set(SENSORS)

    tags = set(ds.get_list_of_tags())
    assert expected_tags == tags

    # The cache does not screw stuff up
    tags = set(ds.get_list_of_tags())
    assert expected_tags == tags


@pytest.mark.parametrize(
    "influxdb", [(SENSORS, "testdb", "root", "root", "sensors")], indirect=True
)
def test_influx_dataset_attrs(influxdb):
    """
    Test expected attributes
    """
    from_ts = dateutil.parser.isoparse("2016-01-01T09:11:00+00:00")
    to_ts = dateutil.parser.isoparse("2016-01-01T10:30:00+00:00")
    tag_list = SENSORS
    config = {
        "type": "TimeSeriesDataset",
        "from_ts": from_ts,
        "to_ts": to_ts,
        "tag_list": tag_list,
    }
    config["data_provider"] = InfluxDataProvider(
        measurement="sensors",
        value_name="Value",
        client=influx_client_from_uri(uri=INFLUXDB_URI, dataframe_client=True),
    )
    dataset = _get_dataset(config)
    assert hasattr(dataset, "get_metadata")

    metadata = dataset.get_metadata()
    assert isinstance(metadata, dict)
