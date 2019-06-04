# -*- coding: utf-8 -*-

import os
import unittest
import logging
import tempfile

import jinja2
from click.testing import CliRunner

from gordo_components import cli
from gordo_components.cli.cli import expand_model, DEFAULT_MODEL_CONFIG
from tests.utils import temp_env_vars

import json

DATA_CONFIG = (
    "{"
    ' "type": "RandomDataset",'
    ' "train_start_date": "2015-01-01T00:00:00+00:00", '
    ' "train_end_date": "2015-06-01T00:00:00+00:00",'
    ' "tags": ["TRC1","TRC2"]'
    "}"
)

MODEL_CONFIG = {"sklearn.decomposition.pca.PCA": {"svd_solver": "auto"}}

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
                MODEL_NAME="model-name",
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
                MODEL_NAME="model-name",
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
                MODEL_NAME="model-name",
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
                MODEL_NAME="model-name",
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
                MODEL_NAME="model-name",
                OUTPUT_DIR=os.path.join(tmpdir, "dir2"),
                # NOTE: Different train dates!
                DATA_CONFIG=(
                    "{"
                    ' "type": "RandomDataset",'
                    ' "train_start_date": "2019-01-01T00:00:00+00:00", '
                    ' "train_end_date": "2019-06-01T00:00:00+00:00",'
                    ' "tags": ["TRC1", "TRC2"]'
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

    def test_build_model_with_parameters(self):
        """
        It works to build a simple model with parameters set
        """

        model = """
        {
         "sklearn.decomposition.pca.PCA":
          {
            "svd_solver": "{{svd_solver}}",
            "n_components": {{n_components}}
          }
        }
        """

        svd_solver = "auto"
        n_components = 0.5

        logger.info(f"MODEL_CONFIG={json.dumps(model)}")

        with tempfile.TemporaryDirectory() as tmpdir:
            with temp_env_vars(
                MODEL_NAME="model-name",
                OUTPUT_DIR=tmpdir,
                DATA_CONFIG=DATA_CONFIG,
                MODEL_CONFIG=model,
            ):
                location_file = f"{os.path.join(tmpdir, 'special-model-location.txt')}"
                args = [
                    "build",
                    "--model-parameter",
                    f"svd_solver,{svd_solver}",
                    "--model-parameter",
                    f"n_components,{n_components}",
                    "--model-location-file",
                    location_file,
                ]

                # Run it twice to ensure the model location in the location file
                # is only written once and not appended.
                for _ in range(2):

                    result = self.runner.invoke(cli.gordo, args=args)

                    self.assertEqual(
                        result.exit_code, 0, msg=f"Command failed: {result}"
                    )
                    self.assertTrue(
                        os.path.exists(location_file),
                        msg=f'Building was supposed to create a model location file at "{location_file}", but it did not!',
                    )
                    with open(location_file, "r") as f:
                        assert f.read() == tmpdir

    def test_expand_model_default_works(self):
        self.assertEquals(expand_model(DEFAULT_MODEL_CONFIG, {}), DEFAULT_MODEL_CONFIG)

    def test_expand_model_expand_works(self):
        model_params = {"kind": "hourglass", "num": 5}
        model_template = "{'gordo_components.model.models.KerasAutoEncoder': {'kind': '{{kind}}', 'num': {{num}}}} "
        expected_model = "{'gordo_components.model.models.KerasAutoEncoder': {'kind': 'hourglass', 'num': 5}} "
        self.assertEquals(expand_model(model_template, model_params), expected_model)

    def test_expand_model_complains_on_missing_vars(self):
        model_params = {"kind": "hourglass"}
        model_template = "{'gordo_components.model.models.KerasAutoEncoder': {'kind': '{{kind}}', 'num': {{num}}}} "
        with self.assertRaises(ValueError):
            expand_model(model_template, model_params)
