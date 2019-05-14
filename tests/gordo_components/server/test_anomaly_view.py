# -*- coding: utf-8 -*-

import pytest
import numpy as np

import tests.utils as tu


@pytest.mark.parametrize(
    "data_to_post",
    [
        {
            "X": np.random.random(size=(10, len(tu.SENSORS_STR_LIST))).tolist()
        },  # Nested records
        {
            "X": np.random.random(size=len(tu.SENSORS_STR_LIST)).tolist()
        },  # Single records 1d
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

    # Verify keys & structure of one output record output contains a list of:
    # {'start': timestamp, 'end': timestamp, 'tags': {'tag': float}, 'total_abnormality': float}
    assert isinstance(resp.json["data"], list)
    record = resp.json["data"][0]
    assert isinstance(record, dict)

    # Only different between POST and GET is POST will return None for
    # start and end dates, because the server can't know what those are
    assert "start" in record
    assert (
        record["start"][0] is None
        if data_to_post is not None
        else isinstance(record["start"][0], str)
    )
    assert "end" in record
    assert (
        record["end"][0] is None
        if data_to_post is not None
        else isinstance(record["end"][0], str)
    )

    assert "total-transformed-error" in record
    assert isinstance(record["total-transformed-error"], list)

    assert "total-untransformed-error" in record
    assert isinstance(record["total-untransformed-error"], list)

    assert "error-transformed" in record
    assert isinstance(record["error-transformed"], list)

    assert "error-untransformed" in record
    assert isinstance(record["error-untransformed"], list)

    assert "original-input" in record
    assert isinstance(record["original-input"], list)

    assert "inverse-transformed-model-output" in record
    assert isinstance(record["inverse-transformed-model-output"], list)

    assert "model-output" in record
    assert isinstance(record["model-output"], list)


def test_more_than_24_hrs(influxdb, gordo_ml_server_client):
    # Request greater than 1 day should be a bad request
    resp = gordo_ml_server_client.get(
        "/anomaly/prediction",
        json={"start": "2016-01-01T00:00:00+00:00", "end": "2016-01-02T00:00:00+00:00"},
    )
    assert resp.status_code == 400

    # and for sanity, less than 1 day are ok
    resp = gordo_ml_server_client.get(
        "/anomaly/prediction",
        json={"start": "2016-01-01T00:00:00+00:00", "end": "2016-01-01T01:00:00+00:00"},
    )
    assert resp.status_code == 200


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
    assert len(resp.json["data"]) == 1, f"Expected one prediction, got: {resp.json}"
    assert resp.json["data"][0]["start"] == ["2016-01-01T00:10:00+00:00"]
    assert resp.json["data"][0]["end"] == ["2016-01-01T00:20:00+00:00"]
