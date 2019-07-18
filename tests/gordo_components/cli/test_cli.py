# -*- coding: utf-8 -*-

import os
import unittest
import pytest
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
    ' "tags": ["TRC1","TRC2"],'
    ' "target_tag_list":["TRC1","TRC2"]'
    "}"
)

MODEL_CONFIG = {"sklearn.decomposition.pca.PCA": {"svd_solver": "auto"}}
MODEL_CONFIG_WITH_PREDICT = {
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
                    ' "tags": ["TRC1", "TRC2"],'
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


@pytest.mark.parametrize(
    "should_be_equal, cv_mode", [(True, "full_build"), (False, "cross_val_only")]
)
def test_build_cv_mode(
    should_be_equal: bool, cv_mode: str, tmp_dir: tempfile.TemporaryDirectory
):
    """
    Testing build with cv_mode set to full and cross_val_only. Checks that cv_scores are
    printed and model are only saved when using the default (full) value.
    """

    runner = CliRunner()

    logger.info(f"MODEL_CONFIG={json.dumps(MODEL_CONFIG_WITH_PREDICT)}")

    with temp_env_vars(
        MODEL_NAME="model-name",
        OUTPUT_DIR=tmp_dir.name,
        DATA_CONFIG=DATA_CONFIG,
        MODEL_CONFIG=json.dumps(MODEL_CONFIG_WITH_PREDICT),
    ):
        location_file = f"{os.path.join(tmp_dir.name, 'location.txt')}"
        result = runner.invoke(
            cli.gordo,
            [
                "build",
                "--print-cv-scores",
                f"--cv-mode={cv_mode}",
                f"--model-location-file={location_file}",
            ],
        )
        # Checks that the file is empty or not depending on the mode.
        if should_be_equal:
            assert os.stat(location_file).st_size != 0
        else:
            assert os.stat(location_file).st_size == 0

        # Checks the output contains 'explained-variance_raw-scores'
        assert "r2-score" in result.output
        assert "mean-squared-error" in result.output
        assert "mean-absolute-error" in result.output
        assert "explained-variance-score" in result.output


@pytest.mark.parametrize(
    "should_be_equal,cv_mode_1, cv_mode_2",
    [
        (True, "full_build", "cross_val_only"),
        (False, "cross_val_only", "cross_val_only"),
    ],
)
def test_build_cv_mode_cross_val_cache(
    should_be_equal: bool,
    cv_mode_1: str,
    cv_mode_2: str,
    tmp_dir: tempfile.TemporaryDirectory,
):
    """
    Checks that cv_scores uses cache if runned after a full build. Loads the same model, and can 
    print the cv_scores from them. 
    """

    runner = CliRunner()

    logger.info(f"MODEL_CONFIG={json.dumps(MODEL_CONFIG)}")

    model_register_dir = f"{os.path.join(tmp_dir.name, 'reg')}"

    with temp_env_vars(
        MODEL_NAME="model-name",
        OUTPUT_DIR=tmp_dir.name,
        DATA_CONFIG=DATA_CONFIG,
        MODEL_CONFIG=json.dumps(MODEL_CONFIG),
        MODEL_REGISTER_DIR=model_register_dir,
    ):

        runner.invoke(cli.gordo, ["build", f"--cv-mode={cv_mode_1}"])
        runner.invoke(cli.gordo, ["build", f"--cv-mode={cv_mode_2}"])

        if should_be_equal:
            assert os.path.exists(model_register_dir)
        else:
            assert not os.path.exists(model_register_dir)


def test_build_cv_mode_build_only(tmp_dir: tempfile.TemporaryDirectory):
    """
    Testing build with cv_mode set to build_only. Checks that the model-location-file exists and
    are not empty. It also checks that the metadata contains cv-duration-sec=None and cv-scores={}
    """

    runner = CliRunner()

    logger.info(f"MODEL_CONFIG={json.dumps(MODEL_CONFIG)}")

    with temp_env_vars(
        MODEL_NAME="model-name",
        OUTPUT_DIR=tmp_dir.name,
        DATA_CONFIG=DATA_CONFIG,
        MODEL_CONFIG=json.dumps(MODEL_CONFIG),
    ):

        location_file = f"{os.path.join(tmp_dir.name, 'location.txt')}"
        metadata_file = f"{os.path.join(tmp_dir.name, 'metadata.json')}"
        runner.invoke(
            cli.gordo,
            ["build", "--cv-mode=build_only", f"--model-location-file={location_file}"],
        )

        assert os.path.exists(location_file)
        assert os.stat(location_file).st_size != 0
        with open(metadata_file) as f:
            metadata_json = json.loads(f.read())
            assert metadata_json["model"]["cross-validation"]["cv-duration-sec"] is None
            assert metadata_json["model"]["cross-validation"]["scores"] == {}
