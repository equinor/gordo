# -*- coding: utf-8 -*-

import tempfile
import typing

import ruamel
import pytest
import numpy as np

from gordo_components import serializer
from tests.test_server.test_gordo_server import SENSORS


@pytest.fixture(scope="session")
def sensors():
    return SENSORS


@pytest.fixture
def tmp_dir():
    """
    Fixture: Temporary directory
    """
    return tempfile.TemporaryDirectory()


@pytest.fixture
def trained_model_directory(tmp_dir, sensors: typing.List[str]):
    """
    Fixture: Train a basic AutoEncoder and save it to a given directory
    will also save some metadata with the model
    """
    definition = ruamel.yaml.load(
        """
        sklearn.pipeline.Pipeline:
            steps:
                - sklearn.preprocessing.data.MinMaxScaler
                - gordo_components.model.models.KerasAutoEncoder:
                    kind: feedforward_hourglass
            memory:
        """,
        Loader=ruamel.yaml.Loader,
    )
    model = serializer.pipeline_from_definition(definition)
    X = np.random.random(size=len(sensors) * 10).reshape(10, len(sensors))
    model.fit(X, X)
    serializer.dump(
        model,
        tmp_dir.name,
        metadata={
            "dataset": {"tag_list": sensors, "resolution": "10T"},
            "user-defined": {"model-name": "test-model", "machine-name": "machine-1"},
        },
    )
    return tmp_dir.name
