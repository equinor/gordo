import unittest
import json
import re

import responses

from gordo_components import __version__
from gordo_components.watchman import server


TARGET_NAMES = ['CT-machine-name-456', 'CT-machine-name-123']
TARGET_NAMES_SANITIZED = ['ct-machine-name-456-kn209d', 'ct-machine-name-123-ksno0s9f092']
PROJECT_NAME = 'some-project-name'
AMBASSADORHOST = 'ambassador'
URL_FORMAT = 'http://{host}/gordo/v0/{project_name}/{sanitized_name}/healthcheck'


def request_callback(_request):
    """
    Mock the Sentinel request to check if a given endpoint is alive or not.
    This imitating a simple /healtcheck endpoint,
    """
    headers = {}
    payload = {'version': __version__}
    return 200, headers, json.dumps(payload)


class WatchmanTestCase(unittest.TestCase):

    def setUp(self):
        app = server.build_app(project_name=PROJECT_NAME, target_names=TARGET_NAMES, target_names_sanitized=TARGET_NAMES_SANITIZED)
        app.testing = True
        self.app = app.test_client()

    def test_healthcheck(self):
        resp = self.app.get('/healthcheck')
        self.assertEqual(resp.status_code, 200)
        resp = resp.get_json()
        self.assertTrue('version' in resp)

    @responses.activate
    def test_api(self):
        """
        Ensure Sentinel API gives a list of expected endpoints and if they are healthy or not.
        """
        # Fake this request; The Sentinel server will start pinging the expected endpoints to see if they are healthy
        # all of which start with the AMBASSADORHOST server; we'll fake these requests.
        responses.add_callback(
            responses.GET, re.compile(rf'.*{AMBASSADORHOST}.*/healthcheck'),
            callback=request_callback,
            content_type='application/json',
        )

        resp = self.app.get('/')
        self.assertEqual(resp.status_code, 200)

        # List of expected endpoints given the current CONFIG_FILE and the project name
        expected_endpoints = [URL_FORMAT.format(host=AMBASSADORHOST,
                                                project_name=PROJECT_NAME,
                                                sanitized_name=sanitized_name)
                              for sanitized_name in TARGET_NAMES_SANITIZED]

        data = resp.get_json()

        # Gives back project name as well.
        self.assertEqual(data['project_name'], PROJECT_NAME)

        for expected, actual in zip(expected_endpoints, data['endpoints']):

            # actual is a dict of {'endpoint': str, 'healthy': bool}
            self.assertEqual(expected.replace(f'http://{AMBASSADORHOST}', ''), actual['endpoint'])
            self.assertTrue(actual['healthy'])
