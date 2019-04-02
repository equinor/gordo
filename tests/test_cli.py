# -*- coding: utf-8 -*-

import os
import unittest
import logging
import tempfile

from click.testing import CliRunner

from gordo_components import cli
from tests.utils import temp_env_vars

import json

DATA_CONFIG = (
    "{"
    ' "type": "RandomDataset",'
    ' "train_start_date": "2015-01-01T00:00:00+00:00", '
    ' "train_end_date": "2015-06-01T00:00:00+00:00",'
    ' "tags": ["tag1", "tag2"]'
    "}"
)

MODEL_CONFIG = {
    "gordo_components.model.models.KerasAutoEncoder": {"kind": "feedforward_hourglass"}
}

logger = logging.getLogger(__name__)


class CliTestCase(unittest.TestCase):
    """
    Test the expected usability of the CLI interface
    """

    def setUp(self):
        self.runner = CliRunner()

    def test_build_env_args(self):
        """
        Instead of passing OUTPUT_DIR directly to CLI, should be able to
        read environment variables
        """

        logger.info(f"MODEL_CONFIG={json.dumps(MODEL_CONFIG)}")

        with tempfile.TemporaryDirectory() as tmpdir:
            with temp_env_vars(
                OUTPUT_DIR=tmpdir,
                DATA_CONFIG=DATA_CONFIG,
                MODEL_CONFIG=json.dumps(MODEL_CONFIG),
            ):
                result = self.runner.invoke(cli.gordo, ["build"])

            self.assertEqual(result.exit_code, 0, msg=f"Command failed: {result}")
            self.assertTrue(
                os.path.exists("/tmp/model-location.txt"),
                msg='Building was supposed to create a "model-location.txt", but it did not!',
            )

    def test_build_use_registry(self):
        """
        Using a registry causes the second build of a model to return the path to the
        first.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            with temp_env_vars(
                OUTPUT_DIR=os.path.join(tmpdir, "dir1"),
                DATA_CONFIG=DATA_CONFIG,
                MODEL_CONFIG=json.dumps(MODEL_CONFIG),
                MODEL_REGISTER_DIR=tmpdir + "/reg",
            ):
                result1 = self.runner.invoke(cli.gordo, ["build"])

            self.assertEqual(result1.exit_code, 0, msg=f"Command failed: {result1}")
            with open("/tmp/model-location.txt") as f:
                first_path = f.read()

            # OUTPUT_DIR is the only difference
            with temp_env_vars(
                OUTPUT_DIR=os.path.join(tmpdir, "dir2"),
                DATA_CONFIG=DATA_CONFIG,
                MODEL_CONFIG=json.dumps(MODEL_CONFIG),
                MODEL_REGISTER_DIR=tmpdir + "/reg",
            ):
                result2 = self.runner.invoke(cli.gordo, ["build"])
            self.assertEqual(result2.exit_code, 0, msg=f"Command failed: {result2}")
            with open("/tmp/model-location.txt") as f:
                second_path = f.read()
            assert first_path == second_path

    def test_build_use_registry_bust_cache(self):
        """
        Even using a registry we get separate model-paths when we ask for models for
        different configurations.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            with temp_env_vars(
                OUTPUT_DIR=os.path.join(tmpdir, "dir1"),
                DATA_CONFIG=DATA_CONFIG,
                MODEL_CONFIG=json.dumps(MODEL_CONFIG),
                MODEL_REGISTER_DIR=tmpdir + "/reg",
            ):
                result1 = self.runner.invoke(cli.gordo, ["build"])

            self.assertEqual(result1.exit_code, 0, msg=f"Command failed: {result1}")
            with open("/tmp/model-location.txt") as f:
                first_path = f.read()

            with temp_env_vars(
                OUTPUT_DIR=os.path.join(tmpdir, "dir2"),
                # NOTE: Different train dates!
                DATA_CONFIG=(
                    "{"
                    ' "type": "RandomDataset",'
                    ' "train_start_date": "2019-01-01T00:00:00+00:00", '
                    ' "train_end_date": "2019-06-01T00:00:00+00:00",'
                    ' "tags": ["tag1", "tag2"]'
                    "}"
                ),
                MODEL_CONFIG=json.dumps(MODEL_CONFIG),
                MODEL_REGISTER_DIR=tmpdir + "/reg",
            ):
                result2 = self.runner.invoke(cli.gordo, ["build"])
            self.assertEqual(result2.exit_code, 0, msg=f"Command failed: {result2}")
            with open("/tmp/model-location.txt") as f:
                second_path = f.read()
            assert first_path != second_path
