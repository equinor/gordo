# -*- coding: utf-8 -*-

import os
import unittest
import pytest
import logging
import tempfile

from click.testing import CliRunner
from unittest import mock


from gordo_components import cli
from gordo_components.cli.cli import expand_model, DEFAULT_MODEL_CONFIG
from gordo_components.serializer import serializer
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
            self.assertGreater(
                len(os.listdir(tmpdir)),
                1,
                msg="Building was supposed to create at least two files (model and "
                "metadata) in OUTPUT_DIR, but it did not!",
            )

    def test_build_use_registry(self):
        """
        Using a registry causes the second build of a model to copy the first to the
        new location.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir_1 = os.path.join(tmpdir, "dir1")
            output_dir_2 = os.path.join(tmpdir, "dir2")

            with temp_env_vars(
                MODEL_NAME="model-name",
                OUTPUT_DIR=output_dir_1,
                DATA_CONFIG=DATA_CONFIG,
                MODEL_CONFIG=json.dumps(MODEL_CONFIG),
                MODEL_REGISTER_DIR=tmpdir + "/reg",
            ):
                result1 = self.runner.invoke(cli.gordo, ["build"])

            self.assertEqual(result1.exit_code, 0, msg=f"Command failed: {result1}")
            # OUTPUT_DIR is the only difference
            with temp_env_vars(
                MODEL_NAME="model-name",
                OUTPUT_DIR=output_dir_2,
                DATA_CONFIG=DATA_CONFIG,
                MODEL_CONFIG=json.dumps(MODEL_CONFIG),
                MODEL_REGISTER_DIR=tmpdir + "/reg",
            ):
                result2 = self.runner.invoke(cli.gordo, ["build"])
            self.assertEqual(result2.exit_code, 0, msg=f"Command failed: {result2}")

            first_metadata = serializer.load_metadata(output_dir_1)
            second_metadata = serializer.load_metadata(output_dir_2)

            # The metadata contains the model build date, so if it got rebuilt these two
            # would be different
            assert (
                first_metadata["model"]["model-creation-date"]
                == second_metadata["model"]["model-creation-date"]
            )

    def test_build_use_registry_bust_cache(self):
        """
        Even using a registry we get separate model-paths when we ask for models for
        different configurations.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir_1 = os.path.join(tmpdir, "dir1")
            output_dir_2 = os.path.join(tmpdir, "dir2")

            with temp_env_vars(
                MODEL_NAME="model-name",
                OUTPUT_DIR=output_dir_1,
                DATA_CONFIG=DATA_CONFIG,
                MODEL_CONFIG=json.dumps(MODEL_CONFIG),
                MODEL_REGISTER_DIR=tmpdir + "/reg",
            ):
                result1 = self.runner.invoke(cli.gordo, ["build"])

            self.assertEqual(result1.exit_code, 0, msg=f"Command failed: {result1}")

            with temp_env_vars(
                MODEL_NAME="model-name",
                OUTPUT_DIR=output_dir_2,
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

            first_metadata = serializer.load_metadata(output_dir_1)
            second_metadata = serializer.load_metadata(output_dir_2)
            # The metadata contains the model build date, so if it got rebuilt these two
            # would be different
            assert (
                first_metadata["model"]["model-creation-date"]
                != second_metadata["model"]["model-creation-date"]
            )

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
                args = [
                    "build",
                    "--model-parameter",
                    f"svd_solver,{svd_solver}",
                    "--model-parameter",
                    f"n_components,{n_components}",
                ]

                # Run it twice to ensure the model location in the location file
                # is only written once and not appended.
                for _ in range(2):

                    result = self.runner.invoke(cli.gordo, args=args)

                    self.assertEqual(
                        result.exit_code, 0, msg=f"Command failed: {result}"
                    )
                    self.assertGreater(
                        len(os.listdir(tmpdir)),
                        1,
                        msg="Building was supposed to create at least two files ("
                        "model and metadata) in OUTPUT_DIR, but it did not!",
                    )

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
    "should_save_model, cv_mode",
    [(True, {"cv_mode": "full_build"}), (False, {"cv_mode": "cross_val_only"})],
)
def test_build_cv_mode(
    should_save_model: bool, cv_mode: str, tmp_dir: tempfile.TemporaryDirectory
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
        result = runner.invoke(
            cli.gordo, ["build", "--print-cv-scores", f"--evaluation-config={cv_mode}"]
        )
        # Checks that the file is empty or not depending on the mode.
        if should_save_model:
            assert len(os.listdir(tmp_dir.name)) != 0
        else:
            assert len(os.listdir(tmp_dir.name)) == 0

        # Checks the output contains 'explained-variance_raw-scores'
        assert "r2-score" in result.output
        assert "mean-squared-error" in result.output
        assert "mean-absolute-error" in result.output
        assert "explained-variance-score" in result.output


@pytest.mark.parametrize(
    "should_be_equal,cv_mode_1, cv_mode_2",
    [
        (True, {"cv_mode": "full_build"}, {"cv_mode": "cross_val_only"}),
        (False, {"cv_mode": "cross_val_only"}, {"cv_mode": "cross_val_only"}),
    ],
)
def test_build_cv_mode_cross_val_cache(
    should_be_equal: bool,
    cv_mode_1: str,
    cv_mode_2: str,
    tmp_dir: tempfile.TemporaryDirectory,
):
    """
    Checks that cv_scores uses cache if ran after a full build. Loads the same model, and can
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

        runner.invoke(cli.gordo, ["build", f"--evaluation-config={cv_mode_1}"])
        runner.invoke(cli.gordo, ["build", f"--evaluation-config={cv_mode_2}"])

        if should_be_equal:
            assert os.path.exists(model_register_dir)
        else:
            assert not os.path.exists(model_register_dir)


def test_build_cv_mode_build_only(tmp_dir: tempfile.TemporaryDirectory):
    """
    Testing build with cv_mode set to build_only. Checks that OUTPUT_DIR gets a model
    saved to it. It also checks that the metadata contains cv-duration-sec=None and
    cv-scores={}
    """

    runner = CliRunner()

    logger.info(f"MODEL_CONFIG={json.dumps(MODEL_CONFIG)}")

    with temp_env_vars(
        MODEL_NAME="model-name",
        OUTPUT_DIR=tmp_dir.name,
        DATA_CONFIG=DATA_CONFIG,
        MODEL_CONFIG=json.dumps(MODEL_CONFIG),
    ):

        metadata_file = f"{os.path.join(tmp_dir.name, 'metadata.json')}"
        runner.invoke(
            cli.gordo, ["build", '--evaluation-config={"cv_mode": "build_only"}']
        )

        # A model has been saved
        assert len(os.listdir(tmp_dir.name)) != 0
        with open(metadata_file) as f:
            metadata_json = json.loads(f.read())
            assert metadata_json["model"]["cross-validation"]["cv-duration-sec"] is None
            assert metadata_json["model"]["cross-validation"]["scores"] == {}


@pytest.mark.parametrize(
    "arg,value,exception_expected",
    [
        # Valid values
        ("--host", "0.0.0.0", False),
        ("--host", "127.0.0.0", False),
        ("--port", 5555, False),
        ("--workers", 1, False),
        ("--log-level", "info", False),
        ("--log-level", "debug", False),
        # Invalid values
        ("--host", "0.0.0", True),
        ("--port", 0, True),
        ("--port", 70000, True),
        ("--workers", -1, True),
        ("--log-level", "badlevel", True),
    ],
)
def test_gunicorn_execution_hosts(monkeypatch, arg, value, exception_expected):
    """
    Test the validation of input parameters to the `run_server` function via the gordo cli
    """

    runner = CliRunner()
    with mock.patch(
        "gordo_components.server.server.run_server",
        mock.MagicMock(return_value=None, autospec=True),
    ) as m:
        result = runner.invoke(cli.gordo, ["run-server", arg, value])

        assert (
            (result.exit_code != 0) if exception_expected else (result.exit_code == 0)
        )
        assert m.called_once_with(value)


def test_watchman_to_sql_cli():
    runner = CliRunner()
    args = ["workflow", "watchman-to-sql", "--help"]
    result = runner.invoke(cli.gordo, args)
    assert result.exit_code == 0
