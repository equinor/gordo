import unittest
import json
import re

import responses

import tests.utils as tu

from gordo_components import __version__
from gordo_components.watchman import server
from tests.mocking.k8s_mocking import mocked_kubernetes


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
            project_name=tu.GORDO_PROJECT,
            project_version=PROJECT_VERSION,
            target_names=tu.GORDO_TARGETS,
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
                project_name=tu.GORDO_PROJECT,
                target_name=target_name,
            )
            for target_name in tu.GORDO_TARGETS
        ]

        data = resp.get_json()

        # Gives back project name as well.
        self.assertEqual(data["project-name"], tu.GORDO_PROJECT)

        for expected, actual in zip(expected_endpoints, data["endpoints"]):

            # actual is a dict of {'endpoint': str, 'healthy': bool}
            self.assertEqual(
                expected.replace(f"http://{AMBASSADOR_NAMESPACE}", ""),
                actual["endpoint"],
            )
            self.assertTrue(actual["healthy"])
            self.assertTrue("endpoint-metadata" in actual)
            self.assertTrue(isinstance(actual["endpoint-metadata"], dict))
