# -*- coding: utf-8 -*-

import os
import logging
import pytest
import subprocess
import shutil
import json

from typing import List
from unittest.mock import patch, MagicMock

from gordo.server.server import run_cmd
from gordo import serializer, __version__
from gordo.server import server

from prometheus_client.registry import CollectorRegistry

import tests.utils as tu


logger = logging.getLogger(__name__)


def test_healthcheck_endpoint(base_route, gordo_ml_server_client):
    """
    Test expected behavior of /<gordo-name>/healthcheck
    """
    # Should also be at the very lowest level as well.
    resp = gordo_ml_server_client.get(f"/healthcheck")
    assert resp.status_code == 200

    resp = gordo_ml_server_client.get(f"{base_route}/healthcheck")
    assert resp.status_code == 200

    data = resp.get_json()
    logger.debug(f"Got resulting JSON response: {data}")
    assert "gordo-server-version" in data


def test_response_header_timing(base_route, gordo_ml_server_client):
    """
    Test that the response contains a `Server-Timing` header
    """
    resp = gordo_ml_server_client.get(f"{base_route}/healthcheck")
    assert resp.status_code == 200
    assert "Server-Timing" in resp.headers
    assert "request_walltime_s" in resp.headers["Server-Timing"]


def test_metadata_endpoint(base_route, gordo_single_target, gordo_ml_server_client):
    """
    Test the expected behavior of /metadata
    """
    resp = gordo_ml_server_client.get(f"{base_route}/metadata")
    assert resp.status_code == 200

    data = resp.get_json()
    assert "metadata" in data
    assert data["metadata"]["name"] == gordo_single_target


def test_download_model(api_version, gordo_project, gordo_name, gordo_ml_server_client):
    """
    Test we can download a model, loadable via serializer.loads()
    """
    resp = gordo_ml_server_client.get(
        f"/gordo/{api_version}/{gordo_project}/{gordo_name}/download-model"
    )

    serialized_model = resp.get_data()
    model = serializer.loads(serialized_model)

    # All models have a fit method
    assert hasattr(model, "fit")

    # Models MUST have either predict or transform
    assert hasattr(model, "predict") or hasattr(model, "transform")

    # Asking for a model that doesn't exist gives 404
    resp = gordo_ml_server_client.get(
        f"/gordo/{api_version}/{gordo_project}/invalid-model-name/download-model"
    )
    assert resp.status_code == 404


def test_run_cmd(monkeypatch):
    """
    Test that execution error catchings work as expected
    """

    # Call command that raises FileNotFoundError, a subclass of OSError
    cmd = ["gumikorn", "gordo.server.server:app"]
    with pytest.raises(OSError):
        run_cmd(cmd)

    # Call command that raises a CalledProcessError
    cmd = ["ping", "--bad-option"]
    with pytest.raises(subprocess.CalledProcessError):
        run_cmd(cmd)


def test_run_server_gthread():
    with patch(
        "gordo.server.server.run_cmd", MagicMock(return_value=None, autospec=True)
    ) as m:
        server.run_server(
            "127.0.0.1",
            9000,
            2,
            "debug",
            worker_connections=50,
            threads=8,
            worker_class="gthread",
        )
        m.assert_called_once_with(
            [
                "gunicorn",
                "--bind",
                "127.0.0.1:9000",
                "--log-level",
                "debug",
                "--error-logfile",
                "-",
                "--access-logfile",
                "-",
                "--worker-class",
                "gthread",
                "--worker-tmp-dir",
                "/dev/shm",
                "--workers",
                "2",
                "--threads",
                "8",
                "gordo.server.server:build_app()",
            ]
        )


def test_run_server_gevent():
    with patch(
        "gordo.server.server.run_cmd", MagicMock(return_value=None, autospec=True)
    ) as m:
        server.run_server(
            "127.0.0.1",
            9000,
            2,
            "debug",
            worker_connections=50,
            threads=8,
            worker_class="gevent",
        )
        m.assert_called_once_with(
            [
                "gunicorn",
                "--bind",
                "127.0.0.1:9000",
                "--log-level",
                "debug",
                "--error-logfile",
                "-",
                "--access-logfile",
                "-",
                "--worker-class",
                "gevent",
                "--worker-tmp-dir",
                "/dev/shm",
                "--workers",
                "2",
                "--worker-connections",
                "50",
                "gordo.server.server:build_app()",
            ]
        )


@pytest.mark.parametrize("revisions", [("1234", "2345", "3456"), ("1234",)])
def test_list_revisions(tmpdir, revisions: List[str]):
    """
    Verify the server is capable of returning the project revisions
    it's capable of serving.
    """

    # Server gets the 'latest' directory to serve models from, but knows other
    # revisions should be available a step up from this directory.
    model_dir = os.path.join(tmpdir, revisions[0])

    # Make revision directories under the tmpdir
    [os.mkdir(os.path.join(tmpdir, rev)) for rev in revisions]  # type: ignore

    # Request from the server what revisions it can serve, should match
    with tu.temp_env_vars(MODEL_COLLECTION_DIR=model_dir):
        app = server.build_app({"ENABLE_PROMETHEUS": False})
        app.testing = True
        client = app.test_client()
        resp = client.get("/gordo/v0/test-project/revisions")
        resp_with_revision = client.get(
            f"/gordo/v0/test-project/revisions?revision={revisions[-1]}"
        )

    assert set(resp.json.keys()) == {"latest", "available-revisions", "revision"}
    assert resp.json["latest"] == os.path.basename(model_dir)
    assert resp.json["revision"] == os.path.basename(model_dir)
    assert isinstance(resp.json["available-revisions"], list)
    assert set(resp.json["available-revisions"]) == set(revisions)

    # And the request asking to use a specific revision gives back that revision,
    # but will return the expected latest available
    assert resp_with_revision.json["latest"] == os.path.basename(model_dir)
    assert resp_with_revision.json["revision"] == revisions[-1]


def test_list_revisions_listdir_fail(caplog):
    """
    Verify the server will not fail if listing directories above the current
    model collection directory it has, fails.
    """

    def listdir_fail(*args, **kwargs):
        raise FileNotFoundError()

    expected_revision = "some-project-revision-123"

    with patch.object(os, "listdir", side_effect=listdir_fail) as mocked_listdir:
        with caplog.at_level(logging.CRITICAL):
            with tu.temp_env_vars(MODEL_COLLECTION_DIR=expected_revision):
                app = server.build_app({"ENABLE_PROMETHEUS": False})
                app.testing = True
                client = app.test_client()
                resp = client.get("/gordo/v0/test-project/revisions")

    assert mocked_listdir.called_once()
    assert set(resp.json.keys()) == {"latest", "available-revisions", "revision"}
    assert resp.json["latest"] == expected_revision
    assert isinstance(resp.json["available-revisions"], list)
    assert resp.json["available-revisions"] == [expected_revision]


def test_model_list_view_non_existant_proj():
    with tu.temp_env_vars(MODEL_COLLECTION_DIR=os.path.join("does", "not", "exist")):
        app = server.build_app({"ENABLE_PROMETHEUS": False})
        app.testing = True
        client = app.test_client()
        resp = client.get("/gordo/v0/test-project/models")
        assert resp.status_code == 200
        assert resp.json["models"] == []


@pytest.mark.parametrize(
    "revision_to_models",
    [
        {"123": ("model-1", "model-2"), "456": ("model-3", "model-4")},
        {"123": (), "456": ("model-1",)},
        dict(),
    ],
)
def test_models_by_revision_list_view(caplog, tmpdir, revision_to_models):
    """
    Server returns expected models it can serve under specific revisions.

    revision_to_models: Dict[str, Tuple[str, ...]]
        Map revision codes to models belonging to that revision.
        Simulate serving some revision, but having access to other
        revisions and its models.
    """

    # Current collection dir for the server, order isn't important.
    if revision_to_models:
        collection_dir = os.path.join(tmpdir, list(revision_to_models.keys())[0])
    else:
        # This will cause a failure to look up a certain revision
        collection_dir = str(tmpdir)

    # Make all the revision and model subfolders
    for revision in revision_to_models.keys():
        os.mkdir(os.path.join(tmpdir, revision))
        for model in revision_to_models[revision]:
            os.makedirs(os.path.join(tmpdir, revision, model), exist_ok=True)

    with tu.temp_env_vars(MODEL_COLLECTION_DIR=collection_dir):
        app = server.build_app({"ENABLE_PROMETHEUS": False})
        app.testing = True
        client = app.test_client()
        for revision in revision_to_models:
            resp = client.get(f"/gordo/v0/test-project/models?revision={revision}")
            assert resp.status_code == 200
            assert "models" in resp.json
            assert sorted(resp.json["models"]) == sorted(revision_to_models[revision])
        else:
            # revision_to_models is empty, so there is nothing on the server.
            # Test that asking for some arbitrary revision will give a 404 and error message
            resp = client.get(
                f"/gordo/v0/test-project/models?revision=revision-does-not-exist"
            )
            assert resp.status_code == 410
            assert resp.json == {
                "error": "Revision 'revision-does-not-exist' not found.",
                "revision": "revision-does-not-exist",
            }


@pytest.mark.parametrize("revisions", (("123", "456"), ("123",)))
def test_request_specific_revision(trained_model_directory, tmpdir, revisions):

    model_name = "test-model"
    current_revision = revisions[0]
    collection_dir = os.path.join(tmpdir, current_revision)

    # Copy trained model into revision model folders
    for revision in revisions:
        model_dir = os.path.join(tmpdir, revision, model_name)
        shutil.copytree(trained_model_directory, model_dir)

        # Now overwrite the metadata.json file to ensure the server actually reads
        # the metadata for this specific revision
        metadata_file = os.path.join(model_dir, "metadata.json")
        assert os.path.isfile(metadata_file)
        with open(metadata_file, "w") as fp:
            json.dump({"revision": revision, "model": model_name}, fp)

    with tu.temp_env_vars(MODEL_COLLECTION_DIR=collection_dir):
        app = server.build_app({"ENABLE_PROMETHEUS": False})
        app.testing = True
        client = app.test_client()
        for revision in revisions:
            resp = client.get(
                f"/gordo/v0/test-project/{model_name}/metadata?revision={revision}"
            )
            assert resp.status_code == 200
            assert resp.json["revision"] == revision

            # Verify the server read the metadata.json file we had overwritten
            assert resp.json["metadata"] == {"revision": revision, "model": model_name}

        # Asking for a revision which doesn't exist gives a 410 Gone.
        resp = client.get(
            f"/gordo/v0/test-project/{model_name}/metadata?revision=does-not-exist"
        )
        assert resp.status_code == 410
        assert resp.json == {
            "error": "Revision 'does-not-exist' not found.",
            "revision": "does-not-exist",
        }

        # Again but by setting header, to ensure we also check the header
        resp = client.get(
            f"/gordo/v0/test-project/{model_name}/metadata",
            headers={"revision": "does-not-exist"},
        )
        assert resp.status_code == 410
        assert resp.json == {
            "error": "Revision 'does-not-exist' not found.",
            "revision": "does-not-exist",
        }


def test_server_version_route(model_collection_directory, gordo_revision):
    """
    Simple route which returns the current version
    """
    with tu.temp_env_vars(MODEL_COLLECTION_DIR=model_collection_directory):
        app = server.build_app({"ENABLE_PROMETHEUS": False})
        app.testing = True
        client = app.test_client()

        resp = client.get("/server-version")
        assert resp.status_code == 200
        assert resp.json == {"revision": gordo_revision, "version": __version__}


def test_non_existant_model_metadata(tmpdir, gordo_project, api_version):
    """
    Simple route which returns the current version
    """
    with tu.temp_env_vars(MODEL_COLLECTION_DIR=str(tmpdir)):
        app = server.build_app({"ENABLE_PROMETHEUS": False})
        app.testing = True
        client = app.test_client()

        resp = client.get(
            f"/gordo/{api_version}/{gordo_project}/model-does-not-exist/metadata"
        )
        assert resp.status_code == 404


def test_expected_models_route(tmpdir):
    """
    Route that gives back the expected models names, which are just read from
    the 'EXPECTED_MODELS' env var.
    """
    with tu.temp_env_vars(
        MODEL_COLLECTION_DIR=str(tmpdir),
        EXPECTED_MODELS=json.dumps(["model-a", "model-b"]),
    ):
        app = server.build_app({"ENABLE_PROMETHEUS": False})
        app.testing = True
        client = app.test_client()

        resp = client.get("/gordo/v0/test-project/expected-models")
        assert resp.json["expected-models"] == ["model-a", "model-b"]


def test_with_prometheus():
    prometheus_registry = CollectorRegistry()
    app = server.build_app({"ENABLE_PROMETHEUS": True}, prometheus_registry)
    app.testing = True
    client = app.test_client()

    client.get("/server-version")

    samples = []
    for metric in prometheus_registry.collect():
        for sample in metric.samples:
            if sample.name == "gordo_server_requests_total":
                samples.append(sample)

    assert (
        len(samples) != 0
    ), "Could not found any 'gordo_server_requests_total' metrics"
    assert len(samples) == 1, "Found more then 1 'gordo_server_requests_total' metric"
