# -*- coding: utf-8 -*-

import requests
import logging
from typing import Iterable, Optional
from flask import Flask, jsonify, make_response
from flask.views import MethodView
from concurrent.futures import ThreadPoolExecutor

from gordo_components import __version__


# Will contain a list of endpoints to expected models via Ambassador
# see _load_endpoints()
ENDPOINTS = None
PROJECT_NAME = None
TARGET_NAMES = None


logger = logging.getLogger(__name__)


class WatchmanApi(MethodView):
    """
    API view to list expected endpoints in this project space and report if they
    are up or not.
    """

    @staticmethod
    def _check_endpoint(endpoint: str) -> Optional[dict]:
        """
        Check if a given endpoint is healthy, if so, return its metadata

        Parameters
        ----------
        endpoint: str - Enpoint to check. ie. /gordo/v0/test-project/test-machine

        Returns
        -------
        Optional[dict] - Will return None if endpoint is not healthy
        """
        endpoint = endpoint[1:] if endpoint.startswith("/") else endpoint
        endpoint = endpoint.rstrip("/")
        try:
            if requests.get(
                f"http://ambassador.ambassador/{endpoint}/healthcheck", timeout=2
            ).ok:
                resp = requests.get(
                    f"http://ambassador.ambassador/{endpoint}/metadata", timeout=2
                )
                if resp.ok:  # No reason to suspect it wouldn't be by now
                    return resp.json()
                else:
                    return dict()  # Empty metadata then.
        except Exception as exc:
            logger.error(
                f"Failed to check health of gordo-server: {endpoint} --> Error: {exc}"
            )
        return None  # pedantic mypy

    def get(self):
        with ThreadPoolExecutor(max_workers=25) as executor:
            futures = {
                executor.submit(self._check_endpoint, endpoint): endpoint
                for endpoint in ENDPOINTS
            }

            # List of dicts: [{'endpoint': /path/to/endpoint, 'healthy': bool, 'metadata': dict}]
            results = []
            for f in futures:

                metadata = f.result()

                results.append(
                    {
                        "endpoint": futures[f],
                        "healthy": True if metadata is not None else False,
                        "metadata": metadata,
                    }
                )

        payload = jsonify({"endpoints": results, "project_name": PROJECT_NAME})
        resp = make_response(payload, 200)
        resp.headers["Cache-Control"] = "max-age=0"
        return resp


def healthcheck():
    """
    Return gordo version, route for Watchman server
    """
    payload = jsonify({"version": __version__, "config": TARGET_NAMES})
    return payload, 200


def build_app(project_name: str, target_names: Iterable[str]):
    """
    Build app and any associated routes
    """

    # Precompute list of expected endpoints from config file and other global env
    global ENDPOINTS, PROJECT_NAME, TARGET_NAMES
    ENDPOINTS = [
        f"/gordo/v0/{project_name}/{target_name}/" for target_name in target_names
    ]
    PROJECT_NAME = project_name
    TARGET_NAMES = target_names

    # App and routes
    app = Flask(__name__)
    app.add_url_rule(rule="/healthcheck", view_func=healthcheck, methods=["GET"])
    app.add_url_rule(
        rule="/", view_func=WatchmanApi.as_view("sentinel_api"), methods=["GET"]
    )
    return app


def run_server(
    host: str, port: int, debug: bool, project_name: str, target_names: Iterable[str]
):

    app = build_app(project_name, target_names)
    app.run(host, port, debug=debug)
