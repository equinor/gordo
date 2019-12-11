import os
import pytest
import logging

from click.testing import CliRunner
from unittest import mock
import mlflow


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


@pytest.fixture
def runner(tmp_dir):
    mlflow.set_tracking_uri(f"file:{tmp_dir}")
    yield CliRunner()


def test_build_env_args(runner, tmp_dir):
    """
    Instead of passing OUTPUT_DIR directly to CLI, should be able to
    read environment variables
    """

    logger.info(f"MODEL_CONFIG={json.dumps(MODEL_CONFIG)}")

    with temp_env_vars(
        PROJECT_NAME="project-name",
        MODEL_NAME="model-name",
        OUTPUT_DIR=tmp_dir,
        DATA_CONFIG=DATA_CONFIG,
        MODEL_CONFIG=json.dumps(MODEL_CONFIG),
    ):
        result = runner.invoke(cli.gordo, ["build"])

    assert result.exit_code == 0, f"Command failed: {result}, {result.exception}"
    assert (
        len(os.listdir(tmp_dir)) > 1
    ), "Building was supposed to create at least two files (model and metadata) in OUTPUT_DIR, but it did not!"


def test_build_use_registry(runner, tmp_dir):
    """
    Using a registry causes the second build of a model to copy the first to the
    new location.
    """

    output_dir_1 = os.path.join(tmp_dir, "dir1")
    output_dir_2 = os.path.join(tmp_dir, "dir2")

    with temp_env_vars(
        PROJECT_NAME="project-name",
        MODEL_NAME="model-name",
        OUTPUT_DIR=output_dir_1,
        DATA_CONFIG=DATA_CONFIG,
        MODEL_CONFIG=json.dumps(MODEL_CONFIG),
        MODEL_REGISTER_DIR=tmp_dir + "/reg",
    ):
        result1 = runner.invoke(cli.gordo, ["build"])

    assert result1.exit_code == 0, f"Command failed: {result1}"
    # OUTPUT_DIR is the only difference
    with temp_env_vars(
        PROJECT_NAME="project-name",
        MODEL_NAME="model-name",
        OUTPUT_DIR=output_dir_2,
        DATA_CONFIG=DATA_CONFIG,
        MODEL_CONFIG=json.dumps(MODEL_CONFIG),
        MODEL_REGISTER_DIR=tmp_dir + "/reg",
    ):
        result2 = runner.invoke(cli.gordo, ["build"])
    assert result2.exit_code == 0, f"Command failed: {result2}"

    first_metadata = serializer.load_metadata(output_dir_1)
    second_metadata = serializer.load_metadata(output_dir_2)

    # The metadata contains the model build date, so if it got rebuilt these two
    # would be different
    assert (
        first_metadata["metadata"]["build-metadata"]["model"]["model-creation-date"]
        == second_metadata["metadata"]["build-metadata"]["model"]["model-creation-date"]
    )


def test_build_use_registry_bust_cache(runner, tmp_dir):
    """
    Even using a registry we get separate model-paths when we ask for models for
    different configurations.
    """

    output_dir_1 = os.path.join(tmp_dir, "dir1")
    output_dir_2 = os.path.join(tmp_dir, "dir2")

    with temp_env_vars(
        PROJECT_NAME="project-name",
        MODEL_NAME="model-name",
        OUTPUT_DIR=output_dir_1,
        DATA_CONFIG=DATA_CONFIG,
        MODEL_CONFIG=json.dumps(MODEL_CONFIG),
        MODEL_REGISTER_DIR=tmp_dir + "/reg",
    ):
        result1 = runner.invoke(cli.gordo, ["build"])

    assert result1.exit_code == 0, f"Command failed: {result1}"

    with temp_env_vars(
        PROJECT_NAME="project-name",
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
        MODEL_REGISTER_DIR=tmp_dir + "/reg",
    ):
        result2 = runner.invoke(cli.gordo, ["build"])
    assert result2.exit_code == 0, f"Command failed: {result2}"

    first_metadata = serializer.load_metadata(output_dir_1)
    second_metadata = serializer.load_metadata(output_dir_2)
    # The metadata contains the model build date, so if it got rebuilt these two
    # would be different
    assert (
        first_metadata["metadata"]["build-metadata"]["model"]["model-creation-date"]
        != second_metadata["metadata"]["build-metadata"]["model"]["model-creation-date"]
    )


def test_build_model_with_parameters(runner, tmp_dir):
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

    with temp_env_vars(
        PROJECT_NAME="project-name",
        MODEL_NAME="model-name",
        OUTPUT_DIR=tmp_dir,
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

            result = runner.invoke(cli.gordo, args=args)

            assert (
                result.exit_code == 0
            ), f"Command failed: {result}, {result.exception}"
            assert (
                len(os.listdir(tmp_dir)) > 1
            ), "Building was supposed to create at least two files (model and metadata) in OUTPUT_DIR, but it did not!"


def test_expand_model_default_works():
    assert expand_model(DEFAULT_MODEL_CONFIG, {}) == DEFAULT_MODEL_CONFIG


def test_expand_model_expand_works():
    model_params = {"kind": "hourglass", "num": 5}
    model_template = "{'gordo_components.model.models.KerasAutoEncoder': {'kind': '{{kind}}', 'num': {{num}}}} "
    expected_model = "{'gordo_components.model.models.KerasAutoEncoder': {'kind': 'hourglass', 'num': 5}} "
    assert expand_model(model_template, model_params) == expected_model


def test_expand_model_complains_on_missing_vars():
    model_params = {"kind": "hourglass"}
    model_template = "{'gordo_components.model.models.KerasAutoEncoder': {'kind': '{{kind}}', 'num': {{num}}}} "
    with pytest.raises(ValueError):
        expand_model(model_template, model_params)


@pytest.mark.parametrize(
    "exception,exit_code",  # ArithmeticError is not in the mapping of exceptions to
    # exit codes, so it should default to 1
    [(FileNotFoundError, 30), (Exception, 1), (ArithmeticError, 1)],
)
def test_build_exit_code(exception, exit_code, runner, tmp_dir):
    """
    Test that cli build exists with different exit codes for different errors.
    """

    logger.info(f"MODEL_CONFIG={json.dumps(MODEL_CONFIG_WITH_PREDICT)}")
    with mock.patch(
        "gordo_components.cli.cli.ModelBuilder.build",
        mock.MagicMock(side_effect=exception, autospec=True, return_value=None),
    ):
        with temp_env_vars(
            PROJECT_NAME="project-name",
            MODEL_NAME="model-name",
            OUTPUT_DIR=tmp_dir,
            DATA_CONFIG=DATA_CONFIG,
            MODEL_CONFIG=json.dumps(MODEL_CONFIG_WITH_PREDICT),
        ):
            result = runner.invoke(
                cli.gordo,
                ["build", '--evaluation-config={"cv_mode": "cross_val_only"}'],
            )
            assert result.exit_code == exit_code


@pytest.mark.parametrize(
    "should_save_model, cv_mode",
    [(True, {"cv_mode": "full_build"}), (False, {"cv_mode": "cross_val_only"})],
)
def test_build_cv_mode(
    runner: CliRunner, should_save_model: bool, cv_mode: str, tmp_dir: str
):
    """
    Testing build with cv_mode set to full and cross_val_only. Checks that cv_scores are
    printed and model are only saved when using the default (full) value.
    """
    logger.info(f"MODEL_CONFIG={json.dumps(MODEL_CONFIG_WITH_PREDICT)}")

    tmp_model_dir = os.path.join(tmp_dir, "tmp")
    os.makedirs(tmp_model_dir, exist_ok=True)

    with temp_env_vars(
        PROJECT_NAME="project-name",
        MODEL_NAME="model-name",
        OUTPUT_DIR=tmp_model_dir,
        DATA_CONFIG=DATA_CONFIG,
        MODEL_CONFIG=json.dumps(MODEL_CONFIG_WITH_PREDICT),
    ):
        result = runner.invoke(
            cli.gordo, ["build", "--print-cv-scores", f"--evaluation-config={cv_mode}"]
        )
        assert result.exit_code == 0
        # Checks that the file is empty or not depending on the mode.
        if should_save_model:
            assert len(os.listdir(tmp_model_dir)) != 0
        else:
            assert len(os.listdir(tmp_model_dir)) == 0

        # Checks the output contains 'explained-variance_raw-scores'
        assert "r2-score" in result.output
        assert "mean-squared-error" in result.output
        assert "mean-absolute-error" in result.output
        assert "explained-variance-score" in result.output


@pytest.mark.parametrize(
    "should_save_model, cv_mode_1, cv_mode_2",
    [
        (True, {"cv_mode": "full_build"}, {"cv_mode": "cross_val_only"}),
        (False, {"cv_mode": "cross_val_only"}, {"cv_mode": "cross_val_only"}),
    ],
)
def test_build_cv_mode_cross_val_cache(
    should_save_model: bool,
    cv_mode_1: str,
    cv_mode_2: str,
    runner: CliRunner,
    tmp_dir: str,
):
    """
    Checks that cv_scores uses cache if ran after a full build. Loads the same model, and can
    print the cv_scores from them.
    """
    logger.info(f"MODEL_CONFIG={json.dumps(MODEL_CONFIG)}")

    with temp_env_vars(
        PROJECT_NAME="project-name",
        MODEL_NAME="model-name",
        OUTPUT_DIR=tmp_dir,
        DATA_CONFIG=DATA_CONFIG,
        MODEL_CONFIG=json.dumps(MODEL_CONFIG),
    ):

        runner.invoke(cli.gordo, ["build", f"--evaluation-config={cv_mode_1}"])
        runner.invoke(cli.gordo, ["build", f"--evaluation-config={cv_mode_2}"])

        if should_save_model:
            assert len(os.listdir(tmp_dir)) > 0
        else:
            assert len(os.listdir(tmp_dir)) == 0


def test_build_cv_mode_build_only(runner: CliRunner, tmp_dir: str):
    """
    Testing build with cv_mode set to build_only. Checks that OUTPUT_DIR gets a model
    saved to it. It also checks that the metadata contains cv-duration-sec=None and
    cv-scores={}
    """

    logger.info(f"MODEL_CONFIG={json.dumps(MODEL_CONFIG)}")

    with temp_env_vars(
        PROJECT_NAME="project-name",
        MODEL_NAME="model-name",
        OUTPUT_DIR=tmp_dir,
        DATA_CONFIG=DATA_CONFIG,
        MODEL_CONFIG=json.dumps(MODEL_CONFIG),
    ):

        metadata_file = f"{os.path.join(tmp_dir, 'metadata.json')}"
        runner.invoke(
            cli.gordo, ["build", '--evaluation-config={"cv_mode": "build_only"}']
        )

        # A model has been saved
        assert len(os.listdir(tmp_dir)) != 0
        with open(metadata_file) as f:
            metadata_json = json.loads(f.read())
            assert (
                metadata_json["metadata"]["build-metadata"]["model"][
                    "cross-validation"
                ]["cv-duration-sec"]
                is None
            )
            assert (
                metadata_json["metadata"]["build-metadata"]["model"][
                    "cross-validation"
                ]["scores"]
                == {}
            )


@mock.patch("gordo_components.builder.mlflow_utils.get_spauth_kwargs")
@mock.patch("gordo_components.builder.mlflow_utils.get_workspace_kwargs")
@mock.patch("gordo_components.builder.mlflow_utils.get_mlflow_client")
def test_enable_remote_logging(
    MockClient,
    mock_get_workspace_kwargs,
    mock_get_spauth_kwargs,
    monkeypatch,
    runner,
    tmp_dir,
):
    """
    Tests disabling MlFlow logging in cli, and missing env var when enabled
    """

    mlflow.set_tracking_uri(f"file:{tmp_dir}")
    with temp_env_vars(
        PROJECT_NAME="project-name",
        MODEL_NAME="model-name",
        OUTPUT_DIR=tmp_dir,
        DATA_CONFIG=DATA_CONFIG,
        MODEL_CONFIG=json.dumps(MODEL_CONFIG),
    ):
        # Logging enabled, without env vars set:
        # Raise error
        result = runner.invoke(cli.gordo, ["build", "--enable-remote-logging=True"])
        result.exit_code != 0

        # Logging enabled, with env vars set:
        # Build success, remote logging executed
        with monkeypatch.context() as m:
            m.setenv("DL_SERVICE_AUTH_STR", "test:test:test")
            m.setenv("AZUREML_WORKSPACE_STR", "test:test:test")

            result = runner.invoke(cli.gordo, ["build", "--enable-remote-logging=True"])
            result.exit_code == 1
            assert MockClient.called
            assert mock_get_workspace_kwargs.called
            assert mock_get_spauth_kwargs.called

        # Reset call counts
        for m in [MockClient, mock_get_workspace_kwargs, mock_get_spauth_kwargs]:
            m.reset_mock()

        # Logging not enabled:
        # Build success, remote logging not executed
        result = runner.invoke(cli.gordo, ["build", "--enable-remote-logging=False"])
        result.exit_code == 0
        assert not MockClient.called
        assert not mock_get_workspace_kwargs.called
        assert not mock_get_spauth_kwargs.called


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
def test_gunicorn_execution_hosts(runner, arg, value, exception_expected):
    """
    Test the validation of input parameters to the `run_server` function via the gordo cli
    """

    with mock.patch(
        "gordo_components.server.server.run_server",
        mock.MagicMock(return_value=None, autospec=True),
    ) as m:
        result = runner.invoke(cli.gordo, ["run-server", arg, value])

        assert (
            (result.exit_code != 0) if exception_expected else (result.exit_code == 0)
        )
        assert m.called_once_with(value)


def test_server_to_sql_cli():
    runner = CliRunner()
    args = ["workflow", "server-to-sql", "--help"]
    result = runner.invoke(cli.gordo, args)
    assert result.exit_code == 0
