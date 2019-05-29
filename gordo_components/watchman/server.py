# -*- coding: utf-8 -*-

import os
import logging
from functools import wraps
from typing import Iterable, Optional
from flask import Flask, jsonify, make_response, request, current_app
from flask.views import MethodView
from apscheduler.schedulers.background import BackgroundScheduler

from gordo_components import __version__
from gordo_components.watchman.endpoints_status import EndpointStatuses


logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("WATCHMAN_LOGLEVEL", "INFO").upper())


ENDPOINT_STATUSES = EndpointStatuses()

# Setup the scheduler to update the endpoint statuses, (fired in build_app)
scheduler = BackgroundScheduler()
scheduler.add_job(ENDPOINT_STATUSES.update, trigger="interval", minutes=5)


class WatchmanApi(MethodView):
    """
    API view to list expected endpoints in this project space and report if they
    are up or not.
    """

    def get(self):

        n_logs = int(request.args.get("logs") or 20) if "logs" in request.args else None

        # If _statuses is None, it hasn't been updated yet by the scheduler.
        if ENDPOINT_STATUSES._statuses is None:
            ENDPOINT_STATUSES.update()

        payload = jsonify(
            {
                "endpoints": ENDPOINT_STATUSES.statuses(n_logs),
                "project-name": current_app.config["PROJECT_NAME"],
            }
        )
        resp = make_response(payload, 200)
        resp.headers["Cache-Control"] = "max-age=0"
        return resp


def healthcheck():
    """
    Return gordo version, route for Watchman server
    """
    payload = jsonify(
        {"version": __version__, "config": current_app.config["TARGET_NAMES"]}
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
    target_names: Iterable[str],
    namespace: str,
    ambassador_namespace: Optional[str] = None,
):
    """
    Build app and any associated routes
    """

    endpoints = [
        f"/gordo/v0/{project_name}/{target_name}/" for target_name in target_names
    ]

    # App and routes
    app = Flask(__name__)
    app.config.update(
        ENDPOINTS=endpoints,
        PROJECT_NAME=project_name,
        PROJECT_VERSION=project_version,
        TARGET_NAMES=list(target_names),
        NAMESPACE=namespace,
        AMBASSADOR_NAMESPACE=ambassador_namespace or namespace,
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
    target_names: Iterable[str],
    namespace: str,
    ambassador_namespace: Optional[str] = None,
):
    app = build_app(
        project_name=project_name,
        project_version=project_version,
        target_names=target_names,
        namespace=namespace,
        ambassador_namespace=ambassador_namespace,
    )
    app.run(host, port, debug=debug, threaded=False)
