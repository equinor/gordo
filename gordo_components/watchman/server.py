# -*- coding: utf-8 -*-

import os
import yaml
import ast
import requests
import logging
from flask import Flask, jsonify, make_response
from flask.views import MethodView
from concurrent.futures import ThreadPoolExecutor

from gordo_components import __version__


# Will contain a list of endpoints to expected models via Ambassador
# see _load_endpoints()
ENDPOINTS = None


logger = logging.getLogger(__name__)


class WatchmanApi(MethodView):
    """
    API view to list expected endpoints in this project space and report if they
    are up or not.
    """
    @staticmethod
    def _check_endpoint(endpoint: str):
        endpoint = endpoint[1:] if endpoint.startswith('/') else endpoint
        try:
            return requests.get(f'http://ambassador/{endpoint}', timeout=2).ok
        except Exception as exc:
            logger.error(f'Failed to check health of gordo-server: {endpoint} --> Error: {exc}')
            return False

    def get(self):
        with ThreadPoolExecutor(max_workers=25) as executor:
            futures = {executor.submit(self._check_endpoint, endpoint): endpoint for endpoint in ENDPOINTS}

            # List of dicts: [{'endpoint': /path/to/endpoint, 'healthy': bool}]
            results = [{'endpoint': futures[f], 'healthy': f.result()} for f in futures]

        payload = jsonify({'endpoints': results, 'project_name': os.environ['PROJECT_NAME']})
        resp = make_response(payload, 200)
        resp.headers['Cache-Control'] = 'max-age=0'
        return resp


def healthcheck():
    """
    Return gordo version, route for Watchman server
    """
    payload = jsonify({'version': __version__, 'config': yaml.load(os.environ['TARGET_NAMES'])})
    return payload, 200


def build_app():
    """
    Build app and any associated routes
    """
    global ENDPOINTS
    ENDPOINTS = _load_endpoints()

    app = Flask(__name__)
    app.add_url_rule(rule='/healthcheck', view_func=healthcheck, methods=['GET'])
    app.add_url_rule(rule='/', view_func=WatchmanApi.as_view('sentinel_api'), methods=['GET'])
    return app


def run_server(host: str = '0.0.0.0', port: int = 5555, debug: bool = False):
    app = build_app()
    app.run(host, port, debug=debug)


def _load_endpoints():
    """
    Given the current environment vars of TARGET_NAMES, PROJECT_NAME, AMBASSADORHOST and PORT: build a list
    of pre-computed expected endpoints
    """
    if 'TARGET_NAMES_SANITIZED' not in os.environ or 'TARGET_NAMES' not in os.environ:
        raise EnvironmentError('Need to have TARGET_NAMES_SANITIZED and TARGET_NAMES environment variables set as a'
                               ' list of expected, sanitized and non-sanitized target / machine names.')
    if 'PROJECT_NAME' not in os.environ:
        raise EnvironmentError('Need to have PROJECT_NAME environment variable set.')

    TARGET_NAMES_SANITIZED = ast.literal_eval(os.environ['TARGET_NAMES_SANITIZED'])
    _TARGET_NAMES = ast.literal_eval(os.environ['TARGET_NAMES'])
    project_name = os.environ["PROJECT_NAME"]

    # Precompute list of expected endpoints from config file
    endpoints = [f'/gordo/v0/{project_name}/{sanitized_name}/healthcheck'
                 for sanitized_name in TARGET_NAMES_SANITIZED]
    return endpoints


if __name__ == '__main__':
    run_server()
