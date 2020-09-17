import os
import pytest
import logging
import re

from click.testing import CliRunner
from unittest import mock
import mlflow


from gordo import cli
from gordo.cli.cli import expand_model
from gordo.serializer import serializer
from gordo.machine import Machine
from tests.utils import temp_env_vars

import json

DATA_CONFIG = {
    "type": "RandomDataset",
    "train_start_date": "2015-01-01T00:00:00+00:00",
    "train_end_date": "2015-06-01T00:00:00+00:00",
    "tags": ["TRC1", "TRC2"],
    "target_tag_list": ["TRC1", "TRC2"],
}

DEFAULT_MODEL_CONFIG = {
    "gordo.machine.model.models.KerasAutoEncoder": {"kind": "feedforward_hourglass"}
}


MODEL_CONFIG = {"sklearn.decomposition.PCA": {"svd_solver": "auto"}}
MODEL_CONFIG_WITH_PREDICT = {
    "gordo.machine.model.models.KerasAutoEncoder": {"kind": "feedforward_hourglass"}
}

logger = logging.getLogger(__name__)


@pytest.fixture
def machine():
    return Machine(
        name="test-model",
        model=MODEL_CONFIG,
        dataset=DATA_CONFIG,
        project_name="project-name",
    )


@pytest.fixture
def runner(tmpdir):
    mlflow.set_tracking_uri(f"file:{tmpdir}")
    yield CliRunner()


def test_build_env_args(runner, tmpdir, machine):
    """
    Instead of passing OUTPUT_DIR directly to CLI, should be able to
    read environment variables
    """
    logger.info(f"MODEL_CONFIG={json.dumps(MODEL_CONFIG)}")

    with temp_env_vars(MACHINE=json.dumps(machine.to_dict()), OUTPUT_DIR=str(tmpdir)):
        result = runner.invoke(cli.gordo, ["build"])

    assert result.exit_code == 0, f"Command failed: {result}, {result.exception}"
    assert (
        len(os.listdir(tmpdir)) > 1
    ), "Building was supposed to create at least two files (model and metadata) in OUTPUT_DIR, but it did not!"


def test_build_use_registry(runner, tmpdir, machine):
    """
    Using a registry causes the second build of a model to copy the first to the
    new location.
    """

    output_dir_1 = os.path.join(tmpdir, "dir1")
    output_dir_2 = os.path.join(tmpdir, "dir2")

    with temp_env_vars(
        MACHINE=json.dumps(machine.to_dict()),
        OUTPUT_DIR=output_dir_1,
        MODEL_REGISTER_DIR=os.path.join(tmpdir, "reg"),
    ):
        result1 = runner.invoke(cli.gordo, ["build"])

    assert result1.exit_code == 0, f"Command failed: {result1}"
    # OUTPUT_DIR is the only difference
    with temp_env_vars(
        MACHINE=json.dumps(machine.to_dict()),
        OUTPUT_DIR=output_dir_2,
        MODEL_REGISTER_DIR=os.path.join(tmpdir, "reg"),
    ):
        result2 = runner.invoke(cli.gordo, ["build"])
    assert result2.exit_code == 0, f"Command failed: {result2}"

    first_metadata = serializer.load_metadata(output_dir_1)
    second_metadata = serializer.load_metadata(output_dir_2)

    # The metadata contains the model build date, so if it got rebuilt these two
    # would be different
    assert (
        first_metadata["metadata"]["build_metadata"]["model"]["model_creation_date"]
        == second_metadata["metadata"]["build_metadata"]["model"]["model_creation_date"]
    )


def test_build_use_registry_bust_cache(runner, tmpdir, machine):
    """
    Even using a registry we get separate model-paths when we ask for models for
    different configurations.
    """

    output_dir_1 = os.path.join(tmpdir, "dir1")
    output_dir_2 = os.path.join(tmpdir, "dir2")

    with temp_env_vars(
        MACHINE=json.dumps(machine.to_dict()),
        OUTPUT_DIR=output_dir_1,
        MODEL_REGISTER_DIR=os.path.join(tmpdir, "reg"),
    ):
        result1 = runner.invoke(cli.gordo, ["build"])

    assert result1.exit_code == 0, f"Command failed: {result1}"

    # NOTE: Different train dates!
    machine.dataset = machine.dataset.from_dict(
        {
            "type": "RandomDataset",
            "train_start_date": "2019-01-01T00:00:00+00:00",
            "train_end_date": "2019-06-01T00:00:00+00:00",
            "tags": ["TRC1", "TRC2"],
        }
    )
    with temp_env_vars(
        MACHINE=json.dumps(machine.to_dict()),
        OUTPUT_DIR=output_dir_2,
        MODEL_REGISTER_DIR=os.path.join(tmpdir, "reg"),
    ):
        result2 = runner.invoke(cli.gordo, ["build"])
    assert result2.exit_code == 0, f"Command failed: {result2}"

    first_metadata = serializer.load_metadata(output_dir_1)
    second_metadata = serializer.load_metadata(output_dir_2)
    # The metadata contains the model build date, so if it got rebuilt these two
    # would be different
    assert (
        first_metadata["metadata"]["build_metadata"]["model"]["model_creation_date"]
        != second_metadata["metadata"]["build_metadata"]["model"]["model_creation_date"]
    )


def test_build_model_with_parameters(runner, tmpdir, machine):
    """
    It works to build a simple model with parameters set
    """
    machine._strict = False
    machine.model = """
    {
     "sklearn.decomposition.PCA":
      {
        "svd_solver": "{{svd_solver}}",
        "n_components": {{n_components}}
      }
    }
    """

    svd_solver = "auto"
    n_components = 0.5

    logger.info(f"MODEL_CONFIG={json.dumps(machine.model)}")

    with temp_env_vars(MACHINE=json.dumps(machine.to_dict()), OUTPUT_DIR=str(tmpdir)):
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
                len(os.listdir(tmpdir)) > 1
            ), "Building was supposed to create at least two files (model and metadata) in OUTPUT_DIR, but it did not!"


def test_expand_model_default_works():
    assert expand_model(str(DEFAULT_MODEL_CONFIG), {}) == DEFAULT_MODEL_CONFIG


def test_expand_model_expand_works():
    model_params = {"kind": "hourglass", "num": 5}
    model_template = "{'gordo.machine.model.models.KerasAutoEncoder': {'kind': '{{kind}}', 'num': {{num}}}} "
    expected_model = {
        "gordo.machine.model.models.KerasAutoEncoder": {"kind": "hourglass", "num": 5}
    }
    assert expand_model(model_template, model_params) == expected_model


def test_expand_model_complains_on_missing_vars():
    model_params = {"kind": "hourglass"}
    model_template = "{'gordo.machine.model.models.KerasAutoEncoder': {'kind': '{{kind}}', 'num': {{num}}}} "
    with pytest.raises(ValueError):
        expand_model(model_template, model_params)


@pytest.mark.parametrize(
    "exception,exit_code",  # ArithmeticError is not in the mapping of exceptions to
    # exit codes, so it should default to 1
    [(FileNotFoundError, 30), (Exception, 1), (ArithmeticError, 1)],
)
def test_build_exit_code(exception, exit_code, runner, tmpdir, machine):
    """
    Test that cli build exists with different exit codes for different errors.
    """
    machine.model = MODEL_CONFIG_WITH_PREDICT
    machine.evaluation = {"cv_mode": "cross_val_only"}

    logger.info(f"MODEL_CONFIG={json.dumps(machine.model)}")
    with mock.patch(
        "gordo.cli.cli.ModelBuilder.build",
        mock.MagicMock(side_effect=exception, autospec=True, return_value=None),
    ):
        with temp_env_vars(
            MACHINE=json.dumps(machine.to_dict()), OUTPUT_DIR=str(tmpdir)
        ):
            result = runner.invoke(cli.gordo, ["build"])
            assert result.exit_code == exit_code


@pytest.mark.parametrize(
    "should_save_model, cv_mode",
    [(True, {"cv_mode": "full_build"}), (False, {"cv_mode": "cross_val_only"})],
)
def test_build_cv_mode(
    tmpdir, runner: CliRunner, should_save_model: bool, cv_mode: str, machine: Machine
):
    """
    Testing build with cv_mode set to full and cross_val_only. Checks that cv_scores are
    printed and model are only saved when using the default (full) value.
    """
    machine.model = MODEL_CONFIG_WITH_PREDICT
    machine.evaluation = cv_mode  # type: ignore

    logger.info(f"MODEL_CONFIG={json.dumps(machine.model)}")

    tmp_model_dir = os.path.join(tmpdir, "tmp")
    os.makedirs(tmp_model_dir, exist_ok=True)

    with temp_env_vars(MACHINE=json.dumps(machine.to_dict()), OUTPUT_DIR=tmp_model_dir):
        result = runner.invoke(cli.gordo, ["build", "--print-cv-scores"])
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
    tmpdir,
    should_save_model: bool,
    cv_mode_1: str,
    cv_mode_2: str,
    runner: CliRunner,
    machine: Machine,
):
    """
    Checks that cv_scores uses cache if ran after a full build. Loads the same model, and can
    print the cv_scores from them.
    """
    logger.info(f"MODEL_CONFIG={json.dumps(machine.model)}")

    machine.evaluation = cv_mode_1  # type: ignore
    with temp_env_vars(MACHINE=json.dumps(machine.to_dict()), OUTPUT_DIR=str(tmpdir)):
        runner.invoke(cli.gordo, ["build"])

    machine.evaluation = cv_mode_2  # type: ignore
    with temp_env_vars(MACHINE=json.dumps(machine.to_dict()), OUTPUT_DIR=str(tmpdir)):
        runner.invoke(cli.gordo, ["build"])

    if should_save_model:
        assert len(os.listdir(tmpdir)) > 0
    else:
        assert len(os.listdir(tmpdir)) == 0


def test_build_cv_mode_build_only(tmpdir, runner: CliRunner, machine: Machine):
    """
    Testing build with cv_mode set to build_only. Checks that OUTPUT_DIR gets a model
    saved to it. It also checks that the metadata contains cv-duration-sec=None and
    cv-scores={}
    """

    logger.info(f"MODEL_CONFIG={json.dumps(machine.model)}")
    machine.evaluation = {"cv_mode": "build_only"}

    with temp_env_vars(MACHINE=json.dumps(machine.to_dict()), OUTPUT_DIR=str(tmpdir)):

        metadata_file = f"{os.path.join(tmpdir, 'metadata.json')}"
        runner.invoke(cli.gordo, ["build"])

        # A model has been saved
        assert len(os.listdir(tmpdir)) != 0
        with open(metadata_file) as f:
            metadata_json = json.loads(f.read())
            assert (
                metadata_json["metadata"]["build_metadata"]["model"][
                    "cross_validation"
                ]["cv_duration_sec"]
                is None
            )
            assert (
                metadata_json["metadata"]["build_metadata"]["model"][
                    "cross_validation"
                ]["scores"]
                == {}
            )


@mock.patch("gordo.reporters.mlflow.get_spauth_kwargs")
@mock.patch("gordo.reporters.mlflow.get_workspace_kwargs")
@mock.patch("gordo.reporters.mlflow.get_mlflow_client")
def test_mlflow_reporter_set_cli_build(
    MockClient,
    mock_get_workspace_kwargs,
    mock_get_spauth_kwargs,
    monkeypatch,
    runner,
    tmpdir,
    machine,
):
    """
    Tests disabling MlFlow logging in cli, and missing env var when enabled
    """

    mlflow.set_tracking_uri(f"file:{tmpdir}")
    machine.runtime = dict(
        reporters=[{"gordo.reporters.mlflow.MlFlowReporter": dict()}]
    )

    with temp_env_vars(MACHINE=json.dumps(machine.to_dict()), OUTPUT_DIR=str(tmpdir)):
        # Logging enabled, without env vars set:
        # Raise error
        result = runner.invoke(cli.gordo, ["build"])
        result.exit_code != 0

        # Logging enabled, with env vars set:
        # Build success, remote logging executed
        with monkeypatch.context() as m:
            m.setenv("DL_SERVICE_AUTH_STR", "test:test:test")
            m.setenv("AZUREML_WORKSPACE_STR", "test:test:test")

            result = runner.invoke(cli.gordo, ["build"])
            result.exit_code == 1
            assert MockClient.called
            assert mock_get_workspace_kwargs.called
            assert mock_get_spauth_kwargs.called

        # Reset call counts
        for m in [MockClient, mock_get_workspace_kwargs, mock_get_spauth_kwargs]:
            m.reset_mock()

    # Logging not enabled:
    # Build success, remote logging not executed
    machine.runtime = dict(builder=dict(remote_logging=dict(enable=False)))
    with temp_env_vars(MACHINE=json.dumps(machine.to_dict()), OUTPUT_DIR=str(tmpdir)):
        result = runner.invoke(cli.gordo, ["build"])
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
        ("--threads", 4, False),
        ("--threads", "auto", True),
        ("--worker-class", "gthread", False),
        ("--log-level", "badlevel", True),
    ],
)
def test_gunicorn_execution_hosts(runner, arg, value, exception_expected):
    """
    Test the validation of input parameters to the `run_server` function via the gordo cli
    """

    with mock.patch(
        "gordo.server.server.run_server",
        mock.MagicMock(return_value=None, autospec=True),
    ) as m:
        result = runner.invoke(cli.gordo, ["run-server", arg, value])

        assert (
            (result.exit_code != 0) if exception_expected else (result.exit_code == 0)
        )
        assert m.called_once_with(value)


def test_log_level_cli():
    """
    Test that the --log-level option in the CLI sets the correct log-level in the genreated config file.
    """

    runner = CliRunner()
    args = [
        "--log-level",
        "test_log_level",
        "workflow",
        "generate",
        "--machine-config",
        "examples/config.yaml",
        "--project-name",
        "test",
    ]

    result = runner.invoke(cli.gordo, args)

    # Find the value on the next line after the key GORDO_LOG_LEVEL
    gordo_log_levels = re.findall(
        r"(?<=GORDO_LOG_LEVEL\r|GORDO_LOG_LEVEL\n)[^\r\n]+", result.stdout
    )

    # Assert all the values to the GORDO_LOG_LEVEL key contains the correct log-level
    assert all(["TEST_LOG_LEVEL" in value for value in gordo_log_levels])
