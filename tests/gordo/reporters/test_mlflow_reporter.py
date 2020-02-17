import datetime
import json

import mlflow
from mlflow.entities import Metric, Param
import mock
import pytest

import gordo.reporters.mlflow as mlu
import gordo.machine.machine
from gordo.machine import Machine
from gordo.reporters.exceptions import ReporterException


def test_validate_dict():
    """
    Test that error is raised when a key is missing from the passed dictionary
    """
    required_keys = ["a", "b", "c"]
    # All keys present passes
    assert mlu._validate_dict({"a": 1, "b": 2, "c": 3}, required_keys) is None
    # Key missing fails
    with pytest.raises(ReporterException):
        mlu._validate_dict({"a": 1, "b": 2}, required_keys)


@pytest.mark.parametrize(
    "workspace_kwargs,service_principal_kwargs,n_interactive,n_spauth,n_get_uri",
    [
        # Empty config, local MLflow
        ({}, {}, 0, 0, 0),
        # Config with no auth, Azure Interactive Login
        (
            {
                "subscription_id": "dummy",
                "resource_group": "dummy",
                "workspace_name": "dummy",
            },
            {},
            1,
            0,
            1,
        ),
        # Config with auth, Azure ServicePrincipal Auth
        (
            {
                "subscription_id": "dummy",
                "resource_group": "dummy",
                "workspace_name": "dummy",
            },
            {
                "tenant_id": "dummy",
                "service_principal_id": "dummy",
                "service_principal_password": "dummy",
            },
            0,
            1,
            1,
        ),
    ],
)
def test_get_mlflow_client(
    workspace_kwargs, service_principal_kwargs, n_interactive, n_spauth, n_get_uri
):
    """
    Test that external methods are called correctly given different configurations
    """

    with mock.patch("gordo.reporters.mlflow.Workspace") as MockWorkspace, mock.patch(
        "gordo.reporters.mlflow.InteractiveLoginAuthentication"
    ) as MockInteractiveAuth, mock.patch(
        "gordo.reporters.mlflow.ServicePrincipalAuthentication"
    ) as MockSPAuth, mock.patch(
        "gordo.reporters.mlflow.MlflowClient"
    ) as MockClient:
        MockInteractiveAuth.return_value = True
        MockWorkspace.return_value.get_mlflow_tracking_uri.return_value = "test_uri"
        mlu.get_mlflow_client(workspace_kwargs, service_principal_kwargs)
        assert MockInteractiveAuth.call_count == n_interactive
        assert MockSPAuth.call_count == n_spauth
        assert MockWorkspace.call_count == n_get_uri
        assert MockClient.called_once()


@mock.patch("gordo.reporters.mlflow.InteractiveLoginAuthentication")
@mock.patch("gordo.reporters.mlflow.Workspace")
@mock.patch("gordo.reporters.mlflow.MlflowClient")
def test_get_mlflow_client_config(MockClient, MockWorkspace, MockInteractiveLogin):
    """
    Test that error is raised with incomplete kwargs
    """
    # Incomplete workspace kwargs
    with pytest.raises(ReporterException):
        mlu.get_mlflow_client(
            {"subscription_id": "dummy", "workspace_name": "dummy"}, {}
        )

    # Incomplete service principal kwargs
    with pytest.raises(ReporterException):
        mlu.get_mlflow_client(
            {
                "subscription_id": "dummy",
                "resource_group": "dummy",
                "workspace_name": "dummy",
            },
            {"tenant_id": "dummy", "service_principal_password": "dummy"},
        )


@mock.patch("gordo.reporters.mlflow.MlflowClient.get_experiment_by_name")
@mock.patch("gordo.reporters.mlflow.MlflowClient.create_experiment")
@mock.patch("gordo.reporters.mlflow.MlflowClient.create_run")
def test_get_run_id_external_calls(
    mock_create_run, mock_create_experiment, mock_get_experiment, tmpdir
):
    """
    Test logic for creating an experiment if it does not exist to create new runs
    """

    class MockRunInfo:
        def __init__(self, run_id):
            self.run_id = run_id

    class MockRun:
        def __init__(self, run_id):
            self.info = MockRunInfo(run_id)

    class MockExperiment:
        def __init__(self, experiment_id):
            self.experiment_id = experiment_id

    def _test_calls(test_run_id, n_create_exp, n_create_run):
        """Test that number of calls match those specified"""
        run_id = mlu.get_run_id(client, experiment_name, model_key)
        assert mock_get_experiment.call_count == 1
        assert mock_create_experiment.call_count == n_create_exp
        assert mock_create_run.call_count == n_create_run
        assert run_id == test_run_id

    # Dummy test name/IDs
    experiment_name = "test_experiment"
    test_experiment_id = "dummy_exp_id"
    test_run_id = "dummy_run_id"
    model_key = "dummy_model_key"

    mlflow.set_tracking_uri(f"file:{tmpdir}")
    client = mlu.MlflowClient()

    # Experiment exists
    # Create a run with existing experiment_id
    mock_get_experiment.return_value = MockExperiment(test_experiment_id)
    mock_create_experiment.return_value = MockExperiment(test_experiment_id)
    mock_create_run.return_value = MockRun(test_run_id)
    _test_calls(test_run_id, n_create_exp=0, n_create_run=1)

    # Reset call counts
    for m in [mock_get_experiment, mock_create_experiment, mock_create_run]:
        m.call_count = 0

    # Experiment doesn't exist
    # Create an experiment and use its ID to create a run
    mock_get_experiment.return_value = None
    mock_create_experiment.return_value = MockExperiment(test_experiment_id)
    mock_create_run.return_value = MockRun(test_run_id)
    _test_calls(test_run_id, n_create_exp=1, n_create_run=1)


@pytest.mark.parametrize(
    "x,expected",
    [
        (datetime.datetime(1970, 1, 1), 0),
        (datetime.datetime(1970, 1, 1, 0, 0, 1), 1000),
        (datetime.datetime(1970, 1, 1, 0, 0, 0, 1000), 1),
    ],
)
def test_datetime_to_ms_since_epoch(x, expected):
    """
    Test that datetime is correctly converted to ms since Unix epoch
    """
    assert mlu._datetime_to_ms_since_epoch(x) == expected


@pytest.mark.parametrize(
    "n_metrics,n_max_metrics,n_params,n_max_params,expected_n_batches",
    [
        # N max metrics limiting
        (20, 5, 20, 20, 4),
        # N max params limiting
        (20, 20, 20, 5, 4),
        # N max equal
        (20, 5, 20, 5, 4),
    ],
)
def test_batch_log_items(
    n_metrics, n_max_metrics, n_params, n_max_params, expected_n_batches
):
    metrics = ["dummy_metric"] * n_metrics
    params = ["dummy_param"] * n_params

    log_batches = mlu.batch_log_items(metrics, params, n_max_metrics, n_max_params)

    # Make sure that we get the number of batches based on max number of metrics, params allowed
    assert len(log_batches) == expected_n_batches

    # Regardless of batch, each batch kwargs should have the same keys, "metrics" and "params"
    assert all([("metrics" in b) and ("params" in b) for b in log_batches])


def test_get_machine_log_items(metadata):
    """
    Test that dicts are correctly converted to MLflow types or errors raised
    """
    metrics, params = mlu.get_machine_log_items(Machine(**metadata))

    assert all(type(m) == Metric for m in metrics)
    assert all(type(p) == Param for p in params)


@pytest.mark.parametrize(
    "secret_str,keys,keys_valid",
    [
        ("dummy1:dummy2:dummy3", ["key1", "key2", "key3"], True),
        ("dummy1:dummy2:dummy3", ["key1", "key2"], False),
    ],
)
def test_get_kwargs_from_secret(monkeypatch, secret_str, keys, keys_valid):
    """
    Test that service principal kwargs are generated correctly if env var present
    """
    env_var_name = "TEST_SECRET"

    # TEST_SECRET doesn't exist as env var
    with pytest.raises(ReporterException):
        mlu.get_kwargs_from_secret(env_var_name, keys)

    # TEST_SECRET exists as env var
    monkeypatch.setenv(name=env_var_name, value=secret_str)
    if keys_valid:
        kwargs = mlu.get_kwargs_from_secret(env_var_name, keys)
        for key, value in zip(keys, secret_str.split(":")):
            assert kwargs[key] == value
    else:
        with pytest.raises(ReporterException):
            mlu.get_kwargs_from_secret(env_var_name, keys)


def test_workspace_spauth_kwargs():
    """Make sure an error is thrown when env vars not set"""
    with pytest.raises(ReporterException):
        mlu.get_workspace_kwargs()

    with pytest.raises(ReporterException):
        mlu.get_spauth_kwargs()


def test_MachineEncoder(metadata):
    """
    Test that metadata can dump successfully with MachineEncoder
    """
    assert json.dumps(metadata, cls=gordo.machine.machine.MachineEncoder)


@mock.patch("gordo.reporters.mlflow.MlflowClient", autospec=True)
def test_mlflow_context_log_metadata(MockClient, tmpdir, metadata):
    """
    Test that call to wrapped function initiates MLflow logging or throws warning
    """
    metadata = Machine(**metadata)
    mlflow.set_tracking_uri(f"file:{tmpdir}")

    mock_client = MockClient()
    mock_client.log_batch.return_value = "test"

    # Function with a metadata dict returned
    with mlu.mlflow_context("returns metadata", "unique_key", {}, {}) as (
        mlflow_client,
        run_id,
    ):
        mlu.log_machine(mlflow_client, run_id, metadata)

    assert mock_client.log_batch.called


@mock.patch("gordo.reporters.mlflow.MlflowClient", autospec=True)
def test_mlflow_context_log_error(MockClient, metadata):
    """
    Test that an error while logging metadata as an artifact raises MlflowLoggingError
    """
    metadata = Machine(**metadata)
    mock_client = MockClient()
    mock_client.log_artifacts.side_effect = Exception("Some unknown exception!")

    with pytest.raises(mlu.MlflowLoggingError):
        with mlu.mlflow_context("returns metadata", "unique_key", {}, {}) as (
            mlflow_client,
            run_id,
        ):
            mlu.log_machine(mlflow_client, run_id, metadata)
