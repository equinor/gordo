from datetime import datetime

import mlflow
from mlflow.entities import Metric, Param
import mock
import pytest
from pytz import UTC

import gordo_components.builder.mlflow_utils as mlu
from gordo_components.dataset.sensor_tag import SensorTag


@pytest.fixture
def metadata():
    """Example metadata for testing"""
    return {
        "user-defined": {},
        "name": "model-name",
        "dataset": {
            "tag_list": [
                SensorTag(name="Tag 1", asset=None),
                SensorTag(name="Tag 2", asset=None),
            ],
            "target_tag_list": [
                SensorTag(name="Tag 1", asset=None),
                SensorTag(name="Tag 2", asset=None),
            ],
            "train_start_date": datetime(2017, 12, 25, 6, 0, tzinfo=UTC),
            "train_end_date": datetime(2017, 12, 30, 6, 0, tzinfo=UTC),
            "resolution": "10T",
            "filter": "",
            "row_filter_buffer_size": 0,
            "data_provider": {"type": "RandomDataProvider"},
            "asset": None,
        },
        "model": {
            "model-offset": 0,
            "model-creation-date": "2019-11-19 10:55:19.327827+01:00",
            "model-builder-version": "0.39.1.dev39+gee79144.d20191118",
            "model-config": {
                "sklearn.compose.TransformedTargetRegressor": {
                    "regressor": {
                        "sklearn.pipeline.Pipeline": {
                            "steps": [
                                "sklearn.preprocessing.data.MinMaxScaler",
                                {
                                    "gordo_components.model.models.KerasAutoEncoder": {
                                        "kind": "feedforward_hourglass",
                                        "compression_factor": 0.5,
                                        "encoding_layers": 2,
                                        "func": "tanh",
                                        "out_func": "linear",
                                        "epochs": 3,
                                    }
                                },
                            ]
                        }
                    },
                    "transformer": "sklearn.preprocessing.data.MinMaxScaler",
                }
            },
            "data-query-duration-sec": 0.02436208724975586,
            "model-training-duration-sec": 0.8394930362701416,
            "cross-validation": {
                "cv-duration-sec": 3.47721529006958,
                "scores": {
                    "explained-variance-score-Tag-1": {
                        "fold-mean": 0.051862609201927645,
                        "fold-std": 0.228637331704387,
                        "fold-max": 0.29798892173189884,
                        "fold-min": -0.2528015461893074,
                        "fold-1": -0.2528015461893074,
                        "fold-2": 0.11040045206319149,
                        "fold-3": 0.29798892173189884,
                    },
                    "explained-variance-score": {
                        "fold-mean": 0.0478112662857908,
                        "fold-std": 0.1421360442077271,
                        "fold-max": 0.22234102900316643,
                        "fold-min": -0.12581624610561765,
                        "fold-1": -0.12581624610561765,
                        "fold-2": 0.04690901595982361,
                        "fold-3": 0.22234102900316643,
                    },
                    "r2-score-Tag-1": {
                        "fold-mean": -2.6371828429430884,
                        "fold-std": 2.6597484998295946,
                        "fold-max": -0.5211094268414729,
                        "fold-min": -6.388371625565568,
                        "fold-1": -6.388371625565568,
                        "fold-2": -0.5211094268414729,
                        "fold-3": -1.0020674764222246,
                    },
                    "r2-score": {
                        "fold-mean": -2.666124085334415,
                        "fold-std": 1.2640159158021866,
                        "fold-max": -1.5855654684086353,
                        "fold-min": -4.4396507978661095,
                        "fold-1": -4.4396507978661095,
                        "fold-2": -1.9731559897285007,
                        "fold-3": -1.5855654684086353,
                    },
                    "mean-squared-error-Tag-1": {
                        "fold-mean": 1.112789357356103,
                        "fold-std": 0.7905379023062951,
                        "fold-max": 2.209464994153014,
                        "fold-min": 0.3762968865667915,
                        "fold-1": 2.209464994153014,
                        "fold-2": 0.3762968865667915,
                        "fold-3": 0.7526061913485038,
                    },
                    "mean-squared-error": {
                        "fold-mean": 1.0602699916789435,
                        "fold-std": 0.34699786692441703,
                        "fold-max": 1.5184574251830893,
                        "fold-min": 0.6789938636261353,
                        "fold-1": 1.5184574251830893,
                        "fold-2": 0.6789938636261353,
                        "fold-3": 0.9833586862276062,
                    },
                    "mean-absolute-error-Tag-1": {
                        "fold-mean": 0.838600049984478,
                        "fold-std": 0.37949884754447927,
                        "fold-max": 1.3563660334402199,
                        "fold-min": 0.45737388225784836,
                        "fold-1": 1.3563660334402199,
                        "fold-2": 0.45737388225784836,
                        "fold-3": 0.702060234255366,
                    },
                    "mean-absolute-error": {
                        "fold-mean": 0.8507839848911773,
                        "fold-std": 0.16306138711492707,
                        "fold-max": 1.0627475252183525,
                        "fold-min": 0.6661439848292272,
                        "fold-1": 1.0627475252183525,
                        "fold-2": 0.6661439848292272,
                        "fold-3": 0.8234604446259526,
                    },
                },
            },
            "history": {
                "loss": [0.2504836615532805, 0.195994034409523, 0.16305454268967365],
                "accuracy": [0.56497175, 0.53107345, 0.53107345],
                "params": {
                    "batch_size": 32,
                    "epochs": 3,
                    "steps": 23,
                    "samples": 708,
                    "verbose": 0,
                    "do_validation": False,
                    "metrics": ["loss", "accuracy"],
                },
            },
        },
    }


def test_validate_dict():
    """
    Test that error is raised when a key is missing from the passed dictionary
    """
    required_keys = ["a", "b", "c"]
    # All keys present passes
    assert mlu._validate_dict({"a": 1, "b": 2, "c": 3}, required_keys) is None
    # Key missing fails
    with pytest.raises(ValueError):
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

    with mock.patch(
        "gordo_components.builder.mlflow_utils.Workspace"
    ) as MockWorkspace, mock.patch(
        "gordo_components.builder.mlflow_utils.InteractiveLoginAuthentication"
    ) as MockInteractiveAuth, mock.patch(
        "gordo_components.builder.mlflow_utils.ServicePrincipalAuthentication"
    ) as MockSPAuth, mock.patch(
        "gordo_components.builder.mlflow_utils.MlflowClient"
    ) as MockClient:
        MockInteractiveAuth.return_value = True
        MockWorkspace.return_value.get_mlflow_tracking_uri.return_value = "test_uri"
        mlu.get_mlflow_client(workspace_kwargs, service_principal_kwargs)
        assert MockInteractiveAuth.call_count == n_interactive
        assert MockSPAuth.call_count == n_spauth
        assert MockWorkspace.call_count == n_get_uri
        assert MockClient.called_once()


@mock.patch("gordo_components.builder.mlflow_utils.InteractiveLoginAuthentication")
@mock.patch("gordo_components.builder.mlflow_utils.Workspace")
@mock.patch("gordo_components.builder.mlflow_utils.MlflowClient")
def test_get_mlflow_client_config(MockClient, MockWorkspace, MockInteractiveLogin):
    """
    Test that error is raised with incomplete kwargs
    """
    # Incomplete workspace kwargs
    with pytest.raises(ValueError):
        mlu.get_mlflow_client(
            {"subscription_id": "dummy", "workspace_name": "dummy"}, {}
        )

    # Incomplete service principal kwargs
    with pytest.raises(ValueError):
        mlu.get_mlflow_client(
            {
                "subscription_id": "dummy",
                "resource_group": "dummy",
                "workspace_name": "dummy",
            },
            {"tenant_id": "dummy", "service_principal_password": "dummy"},
        )


@mock.patch("gordo_components.builder.mlflow_utils.MlflowClient.get_experiment_by_name")
@mock.patch("gordo_components.builder.mlflow_utils.MlflowClient.create_experiment")
@mock.patch("gordo_components.builder.mlflow_utils.MlflowClient.create_run")
def test_get_run_id_external_calls(
    mock_create_run, mock_create_experiment, mock_get_experiment, tmp_dir
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

    mlflow.set_tracking_uri(f"file:{tmp_dir}")
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
        (datetime(1970, 1, 1), 0),
        (datetime(1970, 1, 1, 0, 0, 1), 1000),
        (datetime(1970, 1, 1, 0, 0, 0, 1000), 1),
    ],
)
def test_datetime_to_ms_since_epoch(x, expected):
    """
    Test that datetime is correctly converted to ms since Unix epoch
    """
    assert mlu._datetime_to_ms_since_epoch(x) == expected


def test_get_batch_kwargs(metadata):
    """
    Test that dicts are correctly converted to MLflow types or errors raised
    """

    def _test_convert_metadata(metadata):
        batch_kwargs = mlu.get_batch_kwargs(metadata)

        assert all(type(m) == Metric for m in batch_kwargs["metrics"])
        assert all(type(p) == Param for p in batch_kwargs["params"])

    # With cross validation and metric scores
    _test_convert_metadata(metadata)

    # With cross validation, no scores
    metadata["model"]["cross-validation"].pop("scores")
    _test_convert_metadata(metadata)

    # no cross validation or scores
    metadata["model"].pop("cross-validation")
    _test_convert_metadata(metadata)


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
    with pytest.raises(ValueError):
        mlu.get_kwargs_from_secret(env_var_name, keys)

    # TEST_SECRET exists as env var
    monkeypatch.setenv(name=env_var_name, value=secret_str)
    if keys_valid:
        kwargs = mlu.get_kwargs_from_secret(env_var_name, keys)
        for key, value in zip(keys, secret_str.split(":")):
            assert kwargs[key] == value
    else:
        with pytest.raises(ValueError):
            mlu.get_kwargs_from_secret(env_var_name, keys)


def test_workspace_spauth_kwargs():
    """Make sure an error is thrown when env vars not set"""
    with pytest.raises(ValueError):
        mlu.get_workspace_kwargs()

    with pytest.raises(ValueError):
        mlu.get_spauth_kwargs()


@mock.patch("gordo_components.builder.mlflow_utils.MlflowClient", autospec=True)
def test_mlflow_context_log_metadata(MockClient, tmp_dir, recwarn, metadata):
    """
    Test that call to wrapped function initiates MLflow logging or throws warning
    """

    mlflow.set_tracking_uri(f"file:{tmp_dir}")

    mock_client = MockClient()
    mock_client.log_batch.return_value = "test"

    # Function with a metadata dict returned
    with mlu.mlflow_context("returns metadata", "unique_key", {}, {}) as (
        mlflow_client,
        run_id,
    ):
        mlu.log_metadata(mlflow_client, run_id, metadata)

    assert mock_client.log_batch.called
