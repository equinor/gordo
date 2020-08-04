# -*- coding: utf-8 -*-

import ast
import json
import logging
from io import StringIO
from datetime import datetime, timezone

import pytest
import yaml

from gordo import __version__
from gordo.machine.dataset.datasets import TimeSeriesDataset
from gordo.machine import Machine
from gordo.workflow.config_elements.normalized_config import NormalizedConfig
from gordo.workflow.workflow_generator.workflow_generator import get_dict_from_yaml


logger = logging.getLogger(__name__)


def test_dataset_from_dict():
    """
    Test ability to create a Dataset from a config element
    """
    element_str = """
        name: ct-23-0002
        dataset:
          resolution: 2T
          tag_list:
            - GRA-YE  -23-0751X.PV
            - GRA-TE  -23-0698.PV
            - GRA-PIT -23-0619B.PV
          train_start_date: 2011-05-20T01:00:04+02:00
          train_end_date: 2018-05-10T15:05:50+02:00
    """
    dataset_config = get_dict_from_yaml(StringIO(element_str))["dataset"]
    dataset = TimeSeriesDataset.from_dict(dataset_config.copy())
    asdict = dataset.to_dict()
    assert asdict["tag_list"] == [
        "GRA-YE  -23-0751X.PV",
        "GRA-TE  -23-0698.PV",
        "GRA-PIT -23-0619B.PV",
    ]
    assert asdict["resolution"] == "2T"
    assert asdict["train_start_date"] == "2011-05-20T01:00:04+02:00"
    assert asdict["train_end_date"] == "2018-05-10T15:05:50+02:00"


def test_dataset_from_config_checks_dates():
    """
    A dataset needs to have train_start_date properly before train_end_date
    """
    element_str = """
        dataset:
          resolution: 2T
          tags:
            - GRA-YE  -23-0751X.PV
            - GRA-TE  -23-0698.PV
            - GRA-PIT -23-0619B.PV
          train_start_date: 2018-05-10T15:05:50+02:00
          train_end_date: 2018-05-10T15:05:50+02:00
    """
    dataset_config = yaml.load(element_str, Loader=yaml.FullLoader)["dataset"]
    with pytest.raises(ValueError):
        TimeSeriesDataset.from_dict(dataset_config)


@pytest.fixture
def default_globals():
    default_globals = dict(NormalizedConfig.DEFAULT_CONFIG_GLOBALS)
    # set something different here so we dont have to change the test every time we
    # change the default runtime parameters
    default_globals["runtime"] = {
        "server": {
            "resources": {
                "requests": {"memory": 1, "cpu": 2},
                "limits": {"memory": 3, "cpu": 4},
            }
        }
    }
    default_globals["dataset"] = {"asset": "global-asset"}
    return default_globals


def test_machine_from_config(default_globals: dict):
    """
    Test ability to create a Machine from a config element.
    """

    element_str = """
        name: ct-23-0001-machine
        data_provider:
          threads: 10
        dataset:
          tags: [GRA-TE  -23-0733.PV, GRA-TT  -23-0719.PV, GRA-YE  -23-0751X.PV]
          target_tag_list: [GRA-TE -123-456]
          train_start_date: 2018-01-01T09:00:30Z
          train_end_date: 2018-01-02T09:00:30Z
        model:
          sklearn.pipeline.Pipeline:
            steps:
              - sklearn.preprocessing.data.MinMaxScaler
              - gordo.machine.model.models.KerasAutoEncoder:
                  kind: feedforward_hourglass
        evaluation:
            scoring_scaler: Null
        metadata:
          id: special-id
    """
    element = get_dict_from_yaml(StringIO(element_str))
    machine = Machine.from_config(
        element, project_name="test-project-name", config_globals=default_globals
    )
    logger.info(f"{machine}")
    assert isinstance(machine, Machine)
    assert len(machine.dataset.tag_list) == 3

    # The metadata of machine should be json serializable
    json.dumps(machine.to_dict()["metadata"])

    # The metadata of machine should be ast.literal_eval-able when cast as a str
    assert (
        ast.literal_eval(str(machine.to_dict()["metadata"]))
        == machine.to_dict()["metadata"]
    )
    # dictionary representation of the machine expected:
    expected = {
        "dataset": {
            "aggregation_methods": "mean",
            "asset": "global-asset",
            "data_provider": {
                "dl_service_auth_str": None,
                "interactive": False,
                "storename": "dataplatformdlsprod",
                "type": "DataLakeProvider",
            },
            "default_asset": None,
            "high_threshold": 50000,
            "interpolation_limit": "8H",
            "interpolation_method": "linear_interpolation",
            "low_threshold": -1000,
            "n_samples_threshold": 0,
            "resolution": "10T",
            "row_filter": "",
            "row_filter_buffer_size": 0,
            "tag_list": [
                "GRA-TE  -23-0733.PV",
                "GRA-TT  -23-0719.PV",
                "GRA-YE  -23-0751X.PV",
            ],
            "target_tag_list": ["GRA-TE -123-456"],
            "train_end_date": "2018-01-02T09:00:30+00:00",
            "train_start_date": "2018-01-01T09:00:30+00:00",
            "type": "TimeSeriesDataset",
        },
        "evaluation": {
            "cv_mode": "full_build",
            "metrics": [
                "explained_variance_score",
                "r2_score",
                "mean_squared_error",
                "mean_absolute_error",
            ],
            "scoring_scaler": None,
        },
        "metadata": {
            "build_metadata": {
                "model": {
                    "cross_validation": {
                        "cv_duration_sec": None,
                        "scores": {},
                        "splits": {},
                    },
                    "model_builder_version": __version__,
                    "model_creation_date": None,
                    "model_meta": {},
                    "model_offset": 0,
                    "model_training_duration_sec": None,
                },
                "dataset": {"query_duration_sec": None, "dataset_meta": {}},
            },
            "user_defined": {
                "global-metadata": {},
                "machine-metadata": {"id": "special-id"},
            },
        },
        "model": {
            "sklearn.pipeline.Pipeline": {
                "steps": [
                    "sklearn.preprocessing.data.MinMaxScaler",
                    {
                        "gordo.machine.model.models.KerasAutoEncoder": {
                            "kind": "feedforward_hourglass"
                        }
                    },
                ]
            }
        },
        "name": "ct-23-0001-machine",
        "project_name": "test-project-name",
        "runtime": {
            "reporters": [],
            "server": {
                "resources": {
                    "limits": {"cpu": 4, "memory": 3},
                    "requests": {"cpu": 2, "memory": 1},
                }
            },
        },
    }
    assert machine.to_dict() == expected


def test_invalid_model(default_globals: dict):
    """
    Test invalid model with 'step' instead of 'steps'
    """
    element_str = """
        name: ct-23-0001-machine
        data_provider:
          threads: 10
        dataset:
          tags: [GRA-TE  -23-0733.PV, GRA-TT  -23-0719.PV, GRA-YE  -23-0751X.PV]
          target_tag_list: [GRA-TE -123-456]
          train_start_date: 2018-01-01T09:00:30Z
          train_end_date: 2018-01-02T09:00:30Z
        model:
          sklearn.pipeline.Pipeline:
            step:
              - sklearn.preprocessing.data.MinMaxScaler
              - gordo.machine.model.models.KerasAutoEncoder:
                  kind: feedforward_hourglass
        evaluation:
            scoring_scaler: Null
        metadata:
          id: special-id
    """
    element = get_dict_from_yaml(StringIO(element_str))
    with pytest.raises(ValueError):
        Machine.from_config(
            element, project_name="test-project-name", config_globals=default_globals
        )
