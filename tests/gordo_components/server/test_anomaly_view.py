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
    ],
)
@pytest.mark.parametrize("resp_format", ("json", "parquet", None))
def test_anomaly_prediction_endpoint(
    base_route, influxdb, gordo_ml_server_client, data_to_post, sensors, resp_format
):
    """
    Anomaly GET and POST responses are the same
    """
    endpoint = f"{base_route}/anomaly/prediction"
    if resp_format is not None:
        endpoint += f"?format={resp_format}"

    resp = gordo_ml_server_client.post(endpoint, json=data_to_post)

    # From here, the response should be (pretty much) the same format from GET or POST
    assert resp.status_code == 200
    if resp_format in (None, "json"):
        assert "data" in resp.json
        data = server_utils.dataframe_from_dict(resp.json["data"])
    else:
        data = server_utils.dataframe_from_parquet_bytes(resp.data)

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
