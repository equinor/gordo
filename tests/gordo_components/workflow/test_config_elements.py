# -*- coding: utf-8 -*-

import ast
import json
import logging
import unittest
from io import StringIO
from datetime import datetime, timezone

import yaml

from gordo_components.workflow.config_elements.dataset import Dataset
from gordo_components.workflow.config_elements.machine import Machine
from gordo_components.workflow.config_elements.normalized_config import NormalizedConfig
from gordo_components.workflow.workflow_generator.workflow_generator import (
    get_dict_from_yaml,
)


logger = logging.getLogger(__name__)


class DatasetConfigElementTestCase(unittest.TestCase):
    def test_from_config(self):
        """
        Test ability to create a Dataset from a config element
        """
        element_str = """
            name: ct-23-0002
            dataset:
              resolution: 2T
              tags:
                - GRA-YE  -23-0751X.PV
                - GRA-TE  -23-0698.PV
                - GRA-PIT -23-0619B.PV
              train_start_date: 2011-05-20T01:00:04+02:00
              train_end_date: 2018-05-10T15:05:50+02:00
        """
        dataset_config = get_dict_from_yaml(StringIO(element_str))["dataset"]
        dataset = Dataset.from_config(dataset_config.copy())
        dataset_config.update({"type": "TimeSeriesDataset"})
        dataset_config["train_start_date"] = dataset_config[
            "train_start_date"
        ].isoformat()
        dataset_config["train_end_date"] = dataset_config["train_end_date"].isoformat()

        # target_tag_list wasn't specified, but should default to the 'tags'
        dataset_config["target_tag_list"] = [
            "GRA-YE  -23-0751X.PV",
            "GRA-TE  -23-0698.PV",
            "GRA-PIT -23-0619B.PV",
        ]
        self.assertEqual(dataset.to_dict(), dataset_config)

    def test_dataset_from_config_checks_dates(self):
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
        with self.assertRaises(ValueError):
            Dataset.from_config(dataset_config)


class MachineConfigElementTestCase(unittest.TestCase):
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

    def test_from_config(self):
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
                  - gordo_components.model.models.KerasAutoEncoder:
                      kind: feedforward_hourglass
            evaluation:
                scoring_scaler: Null
            metadata:
              id: special-id
        """
        self.maxDiff = None
        element = get_dict_from_yaml(StringIO(element_str))
        machine = Machine.from_config(
            element,
            project_name="test-project-name",
            config_globals=self.default_globals,
        )
        logger.info(f"{machine}")
        self.assertIsInstance(machine, Machine)
        self.assertTrue(len(machine.dataset.tags), 3)

        # The metadata of machine should be json serializable
        json.dumps(machine.to_dict()["metadata"])

        # The metadata of machine should be ast.literal_eval-able when cast as a str
        self.assertEqual(
            ast.literal_eval(str(machine.to_dict()["metadata"])),
            machine.to_dict()["metadata"],
        )
        # dictionary representation of the machine expected:
        self.assertEqual(
            machine.to_dict(),
            {
                "name": "ct-23-0001-machine",
                "data_provider": {"type": "DataLakeProvider", "threads": 10},
                "project_name": "test-project-name",
                "dataset": {
                    "type": "TimeSeriesDataset",
                    "tags": [
                        "GRA-TE  -23-0733.PV",
                        "GRA-TT  -23-0719.PV",
                        "GRA-YE  -23-0751X.PV",
                    ],
                    "target_tag_list": ["GRA-TE -123-456"],
                    "train_start_date": datetime(
                        2018, 1, 1, 9, 0, 30, tzinfo=timezone.utc
                    ).isoformat(),
                    "train_end_date": datetime(
                        2018, 1, 2, 9, 0, 30, tzinfo=timezone.utc
                    ).isoformat(),
                    "asset": "global-asset",
                },
                "evaluation": {
                    "cv_mode": "full_build",
                    "scoring_scaler": None,
                    "metrics": [
                        "explained_variance_score",
                        "r2_score",
                        "mean_squared_error",
                        "mean_absolute_error",
                    ],
                },
                "metadata": {
                    "global-metadata": {},
                    "machine-metadata": {"id": "special-id"},
                },
                "model": {
                    "sklearn.pipeline.Pipeline": {
                        "steps": [
                            "sklearn.preprocessing.data.MinMaxScaler",
                            {
                                "gordo_components.model.models.KerasAutoEncoder": {
                                    "kind": "feedforward_hourglass"
                                }
                            },
                        ]
                    }
                },
                "runtime": {
                    "server": {
                        "resources": {
                            "requests": {"memory": 1, "cpu": 2},
                            "limits": {"memory": 3, "cpu": 4},
                        }
                    }
                },
            },
        )

    def test_invalid_model(self):
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
                  - gordo_components.model.models.KerasAutoEncoder:
                      kind: feedforward_hourglass
            evaluation:
                scoring_scaler: Null
            metadata:
              id: special-id
        """
        element = get_dict_from_yaml(StringIO(element_str))
        with self.assertRaises(ValueError):
            Machine.from_config(
                element,
                project_name="test-project-name",
                config_globals=self.default_globals,
            )
