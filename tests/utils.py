import os
import json
import typing
import re
from contextlib import contextmanager

import responses
import requests
from asynctest import mock as async_mock

from gordo_components.dataset.datasets import RandomDataProvider
from gordo_components.watchman import server as watchman_server
from gordo_components.server import server as gordo_ml_server


@contextmanager
def watchman(
    host: str,
    project: str,
    targets: typing.List[str],
    model_location: str,
    namespace: str = "default",
):
    """
    # TODO: This is bananas, make into a proper object with context support?

    Mock a deployed watchman deployment

    Parameters
    ----------
    host: str
        Host watchman should pretend to run on
    project: str
        Project watchman should pretend to care about
    targets:
        Targets watchman should pretend to care about
    model_location: str
        Directory of the model to use in the target(s)
    namespace: str
        Namespace for watchman to make requests in.

    Returns
    -------
    None
    """

    with temp_env_vars(MODEL_LOCATION=model_location):
        # Create a watchman test app
        watchman_app = watchman_server.build_app(
            project_name=project,
            project_version="v123",
            target_names=targets,
            namespace=namespace,
        )
        watchman_app.testing = True
        watchman_app = watchman_app.test_client()

        # Create gordo ml servers
        gordo_server_app = gordo_ml_server.build_app(data_provider=RandomDataProvider())
        gordo_server_app.testing = True
        gordo_server_app = gordo_server_app.test_client()

        def watchman_callback(_request):
            """
            Redirect calls to a gordo endpoint to reflect what the local testing app gives
            """
            headers = {}
            resp = watchman_app.get("/").json
            return 200, headers, json.dumps(resp)

        def gordo_ml_server_callback(request):
            """
            Redirect calls to a gordo server to reflect what the local testing app gives
            will call the correct path (assuminng only single level paths) on the
            gordo app.
            """
            headers = {}
            last_path = request.path_url.split("/")[-1]
            if request.method == "GET":

                # we may have json data being passed
                kwargs = dict()
                if request.body:
                    kwargs["json"] = json.loads(request.body.decode())

                resp = gordo_server_app.get(f"/{last_path}", **kwargs)
                resp = resp.json or resp.data
            elif request.method == "POST":
                resp = gordo_server_app.post(
                    f"/{last_path}", json=json.loads(request.body.decode())
                ).json
            else:
                raise NotImplementedError(
                    f"Request method {request.method} not yet implemented."
                )
            return 200, headers, json.dumps(resp) if not type(resp) == bytes else resp

        with responses.RequestsMock(
            assert_all_requests_are_fired=False
        ) as rsps, async_mock.patch(
            "gordo_components.client.io.fetch_json",
            # Mock the async call to get, but exclude session kwarg
            side_effect=lambda *args, **kwargs: requests.get(
                *args, **{k: v for k, v in kwargs.items() if k != "session"}
            ).json(),
        ), async_mock.patch(
            "gordo_components.client.io.post_json",
            # Mock the async call to post, but exclude session kwarg
            side_effect=lambda *args, **kwargs: requests.post(
                *args, **{k: v for k, v in kwargs.items() if k != "session"}
            ).json(),
        ):

            # Watchman requests
            rsps.add_callback(
                responses.GET,
                re.compile(rf".*{host}.*\/gordo\/v0\/{project}\/$"),
                callback=watchman_callback,
                content_type="application/json",
            )

            # Gordo ML Server requests
            rsps.add_callback(
                responses.GET,
                re.compile(
                    rf".*{namespace}.ambassador.*\/gordo\/v0\/{project}\/.*.\/.*."
                ),
                callback=gordo_ml_server_callback,
                content_type="application/json",
            )
            rsps.add_callback(
                responses.GET,
                re.compile(rf".*{host}.*\/gordo\/v0\/{project}\/.*.\/.*."),
                callback=gordo_ml_server_callback,
                content_type="application/json",
            )
            rsps.add_callback(
                responses.POST,
                re.compile(rf".*{host}.*\/gordo\/v0\/{project}\/.*.\/.*."),
                callback=gordo_ml_server_callback,
                content_type="application/json",
            )

            rsps.add_passthru("http+docker://")  # Docker
            rsps.add_passthru("http://localhost:8086")  # Local influx

            yield


@contextmanager
def temp_env_vars(**kwargs):
    """
    Temporarily set the process environment variables
    """
    _env = os.environ.copy()

    for key in kwargs:
        os.environ[key] = kwargs[key]

    yield

    os.environ.clear()
    os.environ.update(_env)
