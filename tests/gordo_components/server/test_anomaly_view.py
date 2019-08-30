# -*- coding: utf-8 -*-

import pytest
import numpy as np

from gordo_components.server import utils as server_utils
import tests.utils as tu


@pytest.mark.parametrize(
    "data_to_post",
    [
        {
            "X": np.random.random(size=(10, len(tu.SENSORS_STR_LIST))).tolist(),
            "y": np.random.random(size=(10, len(tu.SENSORS_STR_LIST))).tolist(),
        },  # Nested records
        {
            "X": np.random.random(size=(1, len(tu.SENSORS_STR_LIST))).tolist(),
            "y": np.random.random(size=(1, len(tu.SENSORS_STR_LIST))).tolist(),
        },  # Single record
        None,  # No data, use GET
    ],
)
def test_anomaly_prediction_endpoint(
    influxdb, gordo_ml_server_client, data_to_post, sensors
):
    """
    Anomaly GET and POST responses are the same
    """
    if data_to_post is None:
        resp = gordo_ml_server_client.get(
            "/anomaly/prediction",
            json={
                "start": "2016-01-01T00:00:00+00:00",
                "end": "2016-01-01T12:00:00+00:00",
            },
        )
    else:
        resp = gordo_ml_server_client.post("/anomaly/prediction", json=data_to_post)

    # From here, the response should be (pretty much) the same format from GET or POST
    assert resp.status_code == 200
    assert "data" in resp.json

    # Load data into dataframe
    data = server_utils.dataframe_from_dict(resp.json["data"])

    # Only different between POST and GET is POST will return None for
    # start and end dates, because the server can't know what those are
    assert "start" in data
    assert "end" in data
    if data_to_post is not None:
        assert np.all(data["start"].isna())
        assert np.all(data["end"].isna())
    else:
        assert not np.any(data["start"].isna())
        assert not np.any(data["end"].isna())

    assert all(
        key in data
        for key in ("total-anomaly", "tag-anomaly", "model-input", "model-output")
    )


def test_more_than_24_hrs(influxdb, gordo_ml_server_client):
    # Request greater than 1 day should be a bad request
    resp = gordo_ml_server_client.get(
        "/anomaly/prediction",
        json={"start": "2016-01-01T00:00:00+00:00", "end": "2016-01-02T00:00:00+00:00"},
    )
    assert resp.status_code == 400, f"Response content: {resp.data}"

    # and for sanity, less than 1 day are ok
    resp = gordo_ml_server_client.get(
        "/anomaly/prediction",
        json={"start": "2016-01-01T00:00:00+00:00", "end": "2016-01-01T01:00:00+00:00"},
    )
    assert resp.status_code == 200, f"Response content: {resp.data}"


def test_overlapping_time_buckets(influxdb, gordo_ml_server_client):
    """
    Requests for overlapping time sample buckets should only produce
    predictions whose bucket falls completely within the requested
    time rang

    This should give one prediction with start end of
    2016-01-01 00:10:00+00:00 and 2016-01-01 00:20:00+00:00, respectively.
    because it will not go over, ie. giving a bucket from 00:20:00 to 00:30:00
    but can give a bucket which contains data before the requested start date.
    """
    resp = gordo_ml_server_client.get(
        "/anomaly/prediction",
        json={"start": "2016-01-01T00:11:00+00:00", "end": "2016-01-01T00:21:00+00:00"},
    )
    assert resp.status_code == 200
    data = server_utils.dataframe_from_dict(resp.json["data"])

    assert len(data) == 1, f"Expected one prediction, got: {resp.json}"
    assert data["start"].iloc[0].tolist() == ["2016-01-01T00:10:00+00:00"]
    assert data["end"].iloc[0].tolist() == ["2016-01-01T00:20:00+00:00"]
