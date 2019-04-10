# -*- coding: utf-8 -*-

import ast
import json
import logging
import unittest

from datetime import datetime

import ruamel.yaml as yaml
from gordo_components.workflow.config_elements.dataset import Dataset
from gordo_components.workflow.config_elements.machine import Machine
from gordo_components.workflow.workflow_generator import helpers


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
        dataset_config = yaml.load(element_str)["dataset"]
        dataset = Dataset.from_config(dataset_config)
        dataset_config.update({"type": "TimeSeriesDataset"})
        dataset_config["train_start_date"] = dataset_config[
            "train_start_date"
        ].isoformat()
        dataset_config["train_end_date"] = dataset_config["train_end_date"].isoformat()
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
        dataset_config = yaml.load(element_str)["dataset"]
        with self.assertRaises(ValueError):
            Dataset.from_config(dataset_config)


class MachineConfigElementTestCase(unittest.TestCase):
    def test_from_config(self):
        """
        Test ability to create a Machine from a config element.
        """

        element_str = """
            name: ct-23-0001-machine
            dataset:
              tags: [GRA-TE  -23-0733.PV, GRA-TT  -23-0719.PV, GRA-YE  -23-0751X.PV]
              train_start_date: 2018-01-01T09:00:30
              train_end_date: 2018-01-02T09:00:30
            model:
              sklearn.pipeline.Pipeline:
                steps:
                  - sklearn.preprocessing.data.MinMaxScaler
                  - gordo_components.model.models.KerasAutoEncoder:
                      kind: feedforward_hourglass
            metadata:
              id: special-id
        """

        element = yaml.safe_load(element_str)
        machine = Machine.from_config(element)
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
                "dataset": {
                    "type": "TimeSeriesDataset",
                    "tags": [
                        "GRA-TE  -23-0733.PV",
                        "GRA-TT  -23-0719.PV",
                        "GRA-YE  -23-0751X.PV",
                    ],
                    "train_start_date": datetime(2018, 1, 1, 9, 0, 30).isoformat(),
                    "train_end_date": datetime(2018, 1, 2, 9, 0, 30).isoformat(),
                },
                "metadata": {
                    "global-metadata": {},
                    "machine-metadata": {"id": "special-id"},
                    "machine-name": "ct-23-0001-machine",
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
            },
        )

        for tag, tag_sanitized in zip(
            machine.dataset.tags, machine.dataset.sanitized_tags
        ):
            self.assertEqual(tag_sanitized, helpers._get_sanitized_string(tag))
