import unittest
import json
import re

import responses

from gordo_components import __version__
from gordo_components.watchman import server
from tests.mocking.k8s_mocking import mocked_kubernetes


TARGET_NAMES = ["CT-machine-name-456", "CT-machine-name-123"]
PROJECT_NAME = "some-project-name"
PROJECT_VERSION = "1"
AMBASSADOR_NAMESPACE = "kubeflow"
URL_FORMAT = "http://{host}/gordo/v0/{project_name}/{target_name}/"


def healthcheck_request_callback(_request):
    """
    Mock the Watchman request to check if a given endpoint is alive or not.
    This imitating a simple /healtcheck endpoint,
    """
    headers = {}
    payload = {"version": __version__}
    return 200, headers, json.dumps(payload)


def metadata_request_callback(_request):
    """
    Mock the Watchman request to get metadata from a given gordo server
    This imitating a simple /metadata endpoint,
    """
    headers = {}
    payload = {"version": __version__, "metadata": {"model": "test-model"}}
    return 200, headers, json.dumps(payload)


class WatchmanTestCase(unittest.TestCase):
    def setUp(self):
        app = server.build_app(
            project_name=PROJECT_NAME,
            project_version=PROJECT_VERSION,
            target_names=TARGET_NAMES,
            namespace=AMBASSADOR_NAMESPACE,
        )
        app.testing = True
        self.app = app.test_client()

    def test_healthcheck(self):
        resp = self.app.get("/healthcheck")
        self.assertEqual(resp.status_code, 200)
        resp = resp.get_json()
        self.assertTrue("version" in resp)

    @responses.activate
    @mocked_kubernetes()
    def test_api(self, *_args):
        """
        Ensure Sentinel API gives a list of expected endpoints and if they are healthy or not.
        """
        # Fake this request; The watchman server will start pinging the expected endpoints to see if they are healthy
        # all of which start with the AMBASSADORHOST server; we'll fake these requests.
        responses.add_callback(
            responses.GET,
            re.compile(rf".*{AMBASSADOR_NAMESPACE}.*\/healthcheck"),
            callback=healthcheck_request_callback,
            content_type="application/json",
        )

        responses.add_callback(
            responses.GET,
            re.compile(rf".*{AMBASSADOR_NAMESPACE}.*\/metadata"),
            callback=metadata_request_callback,
            content_type="application/json",
        )

        resp = self.app.get("/")
        self.assertEqual(resp.status_code, 200)

        # List of expected endpoints given the current CONFIG_FILE and the project name
        expected_endpoints = [
            URL_FORMAT.format(
                host=AMBASSADOR_NAMESPACE,
                project_name=PROJECT_NAME,
                target_name=target_name,
            )
            for target_name in TARGET_NAMES
        ]

        data = resp.get_json()

        # Gives back project name as well.
        self.assertEqual(data["project-name"], PROJECT_NAME)

        for expected, actual in zip(expected_endpoints, data["endpoints"]):

            # actual is a dict of {'endpoint': str, 'healthy': bool}
            self.assertEqual(
                expected.replace(f"http://{AMBASSADOR_NAMESPACE}", ""),
                actual["endpoint"],
            )
            self.assertTrue(actual["healthy"])
            self.assertTrue("endpoint-metadata" in actual)
            self.assertTrue(isinstance(actual["endpoint-metadata"], dict))

    @mocked_kubernetes()
    def test_gordo_k8s_workflow(self, *_args):
        """
        Test we can construct a workflow representation for Watchman to use
        """
        from gordo_components.watchman.gordo_k8s_interface import list_model_builders

        model_builders = list_model_builders(
            namespace="default", project_name="gordo-test", project_version="1"
        )

        self.assertEqual(len(model_builders), 1)

        self.assertTrue(
            "test-machine-name" in (pod.target_name for pod in model_builders)
        )
        self.assertTrue(all(pod.is_healthy for pod in model_builders))
        self.assertTrue(
            "gordo-test-pod-name-1234" in (pod.name for pod in model_builders)
        )

    @mocked_kubernetes()
    def test_gordo_k8s_service(self, *_args):
        """
        Test we can construct a service representation for Watchman to use
        """
        from gordo_components.watchman.gordo_k8s_interface import Service

        service = Service(
            namespace="default", name="gordoserver-gordo-test-test-machine-name"
        )
        self.assertEqual(len(service), 1)
        self.assertEqual(service.status, 1.0)
