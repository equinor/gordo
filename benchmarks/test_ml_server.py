# -*- coding: utf-8 -*-

from typing import List

import numpy as np

from tests.conftest import gordo_ml_server_client, trained_model_directory, sensors


"""
`benchmark` is a pytest-benchmark fixture: https://pytest-benchmark.readthedocs.io/en/latest/
"""


def single_post_to_ml_server(client, path: str, X: List[List[float]]):
    """Post some data under 'X' given a client and path"""
    resp = client.post(path, json={"X": X})
    return resp


def test_bench_ml_server_anomaly_post(benchmark, gordo_ml_server_client, sensors):
    """Benchmark posting data to the anomaly endpoint"""
    X = np.random.random((100, len(sensors))).tolist()
    resp = benchmark.pedantic(
        single_post_to_ml_server,
        args=(gordo_ml_server_client, "/anomaly/prediction", X),
        iterations=1,
        rounds=100,
    )
    assert resp.status_code == 200


def test_bench_ml_server_base_post(benchmark, gordo_ml_server_client, sensors):
    """Benchmark posting data to the base prediction endpoint"""
    X = np.random.random((100, len(sensors))).tolist()
    resp = benchmark.pedantic(
        single_post_to_ml_server,
        args=(gordo_ml_server_client, "/prediction", X),
        iterations=1,
        rounds=100,
    )
    assert resp.status_code == 200
