import pytest
import yaml
import re

from gordo.machine.loader import (
    load_globals_config,
    load_machine_config,
    load_model_config,
    MachineConfigException,
)


def test_load_globals_config():
    expected_config = {
        "dataset": {
            "tags": ["tag1", "tag2"],
            "train_end_date": "2022-01-15T00:00:00+00:00",
            "train_start_date": "2021-12-25T00:00:00+00:00",
        },
        "model": {
            "gordo.machine.model.anomaly.diff.DiffBasedAnomalyDetector": {
                "base_estimator": {
                    "sklearn.pipeline.Pipeline": {
                        "steps": [
                            "sklearn.preprocessing.MinMaxScaler",
                            {
                                "gordo.machine.model.models.KerasAutoEncoder": {
                                    "kind": "feedforward_hourglass"
                                }
                            },
                        ]
                    }
                }
            }
        },
        "runtime": {"builder": {"resources": {"requests": {"memory": 1000}}}},
        "metadata": {"key1": "value1"},
        "evaluation": {"cv_mode": "no_cv"},
    }
    config_content = """
    dataset: |
      tags:
      - tag1
      - tag2
      train_end_date: '2022-01-15T00:00:00+00:00'
      train_start_date: '2021-12-25T00:00:00+00:00'
    model: |
      gordo.machine.model.anomaly.diff.DiffBasedAnomalyDetector:
        base_estimator:
          sklearn.pipeline.Pipeline:
            steps:
              - sklearn.preprocessing.MinMaxScaler
              - gordo.machine.model.models.KerasAutoEncoder:
                  kind: feedforward_hourglass
    runtime: |
      builder:
        resources:
          requests:
            memory: 1000
    metadata: |
      key1: value1
    evaluation: |
      cv_mode: no_cv
    """
    global_config = load_globals_config(
        yaml.safe_load(config_content), "spec.config.globals"
    )
    assert global_config == expected_config
    # Test for legacy config logic
    config_content = """
    dataset:
      tags:
      - tag1
      - tag2
      train_end_date: '2022-01-15T00:00:00+00:00'
      train_start_date: '2021-12-25T00:00:00+00:00'
    model: |
      gordo.machine.model.anomaly.diff.DiffBasedAnomalyDetector:
        base_estimator:
          sklearn.pipeline.Pipeline:
            steps:
              - sklearn.preprocessing.MinMaxScaler
              - gordo.machine.model.models.KerasAutoEncoder:
                  kind: feedforward_hourglass
    runtime:
      builder:
        resources:
          requests:
            memory: 1000
    metadata:
      key1: value1
    evaluation:
      cv_mode: no_cv
    """
    global_config = load_globals_config(
        yaml.safe_load(config_content), "spec.config.globals"
    )
    config_content = """
    dataset:
      tags:
      - tag1
      - tag2
      train_end_date: '2022-01-15T00:00:00+00:00'
      train_start_date: '2021-12-25T00:00:00+00:00'
    """
    global_config = load_globals_config(
        yaml.safe_load(config_content), "spec.config.globals"
    )
    assert global_config == {
        "dataset": {
            "tags": ["tag1", "tag2"],
            "train_end_date": "2022-01-15T00:00:00+00:00",
            "train_start_date": "2021-12-25T00:00:00+00:00",
        }
    }
    config_content = """
    dataset: |
      tags:
          - tag1
      - tag2
    """
    with pytest.raises(
        MachineConfigException,
        match=re.escape(r"Error loading YAML from 'spec.config.globals.dataset'"),
    ):
        load_globals_config(yaml.safe_load(config_content), "spec.config.globals")


def test_load_machine_config():
    config_content = """
    name: 'model1'
    dataset: |
      tags:
      - tag1
      - tag2
      train_end_date: '2022-01-15T00:00:00+00:00'
      train_start_date: '2021-12-25T00:00:00+00:00'
    """
    machine_config = load_machine_config(
        yaml.safe_load(config_content), "spec.config.machines[0]"
    )
    assert machine_config == {
        "name": "model1",
        "dataset": {
            "tags": ["tag1", "tag2"],
            "train_end_date": "2022-01-15T00:00:00+00:00",
            "train_start_date": "2021-12-25T00:00:00+00:00",
        },
    }
    config_content = """
    dataset: |
      tags:
      - tag1
      - tag2
      train_start_date: '2021-12-25T00:00:00+00:00'
    """
    with pytest.raises(
        MachineConfigException,
        match=re.escape(r"'spec.config.machines[0].name' is empty"),
    ):
        load_machine_config(yaml.safe_load(config_content), "spec.config.machines[0]")


def test_load_model_config():
    config_content = """
    name: 'model1'
    project_name: 'project1'
    dataset: |
      tags:
      - tag1
      - tag2
      train_end_date: '2022-01-15T00:00:00+00:00'
      train_start_date: '2021-12-25T00:00:00+00:00'
    """
    model_config = load_model_config(yaml.safe_load(config_content), "spec.config")
    assert model_config == {
        "name": "model1",
        "project_name": "project1",
        "dataset": {
            "tags": ["tag1", "tag2"],
            "train_end_date": "2022-01-15T00:00:00+00:00",
            "train_start_date": "2021-12-25T00:00:00+00:00",
        },
    }
    config_content = """
    name: 'model1'
    dataset: |
      tags:
      - tag1
      - tag2
      train_start_date: '2021-12-25T00:00:00+00:00'
    """
    with pytest.raises(
        MachineConfigException, match=re.escape(r"'spec.config.project_name' is empty")
    ):
        load_model_config(yaml.safe_load(config_content), "spec.config")
