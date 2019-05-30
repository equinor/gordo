# -*- coding: utf-8 -*-

import os
import logging
from functools import wraps
from typing import List
from flask import Flask, jsonify, make_response, current_app
from flask.views import MethodView
from apscheduler.schedulers.background import BackgroundScheduler

from gordo_components import __version__
from gordo_components.watchman.endpoints_status import EndpointStatuses


logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("WATCHMAN_LOGLEVEL", "INFO").upper())


# Setup the scheduler to update the endpoint statuses
scheduler = BackgroundScheduler()

ENDPOINT_STATUSES = None  # Initialized below


class WatchmanApi(MethodView):
    """
    API view to list expected endpoints in this project space and report if they
    are up or not.
    """

    def get(self):

        payload = jsonify(
            {
                "endpoints": ENDPOINT_STATUSES.statuses(),
                "project-name": current_app.config["PROJECT_NAME"],
                "project-version": current_app.config["PROJECT_VERSION"],
            }
        )
        resp = make_response(payload, 200)
        resp.headers["Cache-Control"] = "max-age=5"
        return resp


def healthcheck():
    """
    Return gordo version, route for Watchman server
    """
    payload = jsonify(
        {
            "version": __version__,
            "config": current_app.config["TARGET_NAMES"],
            "project-name": current_app.config["PROJECT_NAME"],
            "project-version": current_app.config["PROJECT_VERSION"],
        }
    )
    return payload, 200


def run_override(func):
    """
    Wrapper to Flask's app.run function which will first start the scheduler
    and then call the original app.run()
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        scheduler.start()
        return func(*args, **kwargs)

    return wrapper


def build_app(
    project_name: str,
    project_version: str,
    target_names: List[str],
    namespace: str,
    ambassador_host: str,
    listen_to_kubernetes: bool = True,
):
    """
    Build app and any associated routes
    """

    global ENDPOINT_STATUSES
    ENDPOINT_STATUSES = EndpointStatuses(
        scheduler=scheduler,
        project_name=project_name,
        ambassador_host=ambassador_host,
        model_names=target_names,
        project_version=project_version,
        namespace=namespace,
        listen_to_kubernetes=listen_to_kubernetes,
    )

    # App and routes
    app = Flask(__name__)
    app.config.update(
        PROJECT_NAME=project_name,
        PROJECT_VERSION=project_version,
        TARGET_NAMES=list(target_names),
    )
    app.add_url_rule(rule="/healthcheck", view_func=healthcheck, methods=["GET"])
    app.add_url_rule(
        rule="/", view_func=WatchmanApi.as_view("watchman_api"), methods=["GET"]
    )

    # Ensure that calling app.run will start the scheduler
    app.run = run_override(app.run)  # type: ignore

    return app


def run_server(
    host: str,
    port: int,
    debug: bool,
    project_name: str,
    project_version: str,
    target_names: List[str],
    namespace: str,
    ambassador_host: str,
    listen_to_kubernetes: bool = True,
):
    app = build_app(
        project_name=project_name,
        project_version=project_version,
        target_names=target_names,
        namespace=namespace,
        ambassador_host=ambassador_host,
        listen_to_kubernetes=listen_to_kubernetes,
    )
    app.run(host, port, debug=debug, threaded=False)
