# -*- coding: utf-8 -*-
"""
This module contains code for generating the Gordo server Flask application.

Running this module will run the application using Flask's development webserver.
Gunicorn can be used to run the application as `gevent` async workers by using the
:func:`~gordo.server.server.run_server` function.
"""
import os
import json
import logging
import timeit
import typing
import subprocess
from functools import wraps

import yaml
from flask import Flask, g, request, current_app, make_response, jsonify

from typing import Optional, Any, Dict

from gordo.server import views
from gordo.dependencies import configure_once
from gordo import __version__

from prometheus_client import CollectorRegistry
from .prometheus import GordoServerPrometheusMetrics

logger = logging.getLogger(__name__)


def enable_prometheus():
    return os.getenv("ENABLE_PROMETHEUS", "false") != "false"


class Config:
    """Server config"""

    def __init__(self):
        self.MODEL_COLLECTION_DIR_ENV_VAR = "MODEL_COLLECTION_DIR"
        self.EXPECTED_MODELS = yaml.safe_load(os.getenv("EXPECTED_MODELS", "[]"))
        self.ENABLE_PROMETHEUS = enable_prometheus()
        self.PROJECT = os.getenv("PROJECT")


def adapt_proxy_deployment(wsgi_app: typing.Callable) -> typing.Callable:
    """
    Decorator specific to fixing behind-proxy-issues when on Kubernetes and
    using Envoy proxy.

    Parameters
    ----------
    wsgi_app: typing.Callable
        The underlying WSGI application of a flask app, for example

    Notes
    -----
    Special note about deploying behind Ambassador, or prefixed proxy paths in general:

    When deployed on kubernetes/ambassador there is a prefix in-front of the
    server. ie::

        /gordo/v0/some-project-name/some-target

    The server itself only knows about routes to the right of such a prefix:
    such as ``/metadata`` or ``/predictions`` when in reality, the full path is::

        /gordo/v0/some-project-name/some-target/metadata

    This is solved by getting the current application's assigned prefix,
    where ``HTTP_X_ENVOY_ORIGINAL_PATH`` is the *full* path, including the prefix.
    and ``PATH_INFO`` is the actual relative path the server knows about.

    This function wraps the WSGI app itself to map the current full path
    to the assigned route function.

    ie. ``/metadata`` -> metadata route function, by default, but updates
    ``/gordo/v0/some-project-name/some-target/metadata`` -> metadata route function

    Returns
    -------
    Callable

    Example
    -------
    >>> app = Flask(__name__)
    >>> app.wsgi_app = adapt_proxy_deployment(app.wsgi_app)
    """

    @wraps(wsgi_app)
    def wrapper(environ, start_response):

        # Script name can be "/gordo/v0/some-project-name/some-target/metadata"
        script_name = environ.get("HTTP_X_ENVOY_ORIGINAL_PATH", "")
        if script_name:

            # PATH_INFO could be either "/" or some local route such as "/metadata"
            path_info = environ.get("PATH_INFO", "")
            if path_info.rstrip("/"):

                # PATH_INFO must be something like "/metadata" or other local path
                # To figure out the prefix/script_name we remove it from the
                # full HTTP_X_ENVOY_ORIGINAL_PATH, so that something such as
                # /gordo/v0/some-project-name/some-target/metadata, becomes
                # /gordo/v0/some-project-name/some-target/
                script_name = script_name.replace(path_info, "")
            environ["SCRIPT_NAME"] = script_name

            # Now we can just ensure the PATH_INFO reflects the locally known path
            # such as /metadata and not /gordo/v0/some-project-name/some-target/metadata
            if path_info.startswith(script_name):
                environ["PATH_INFO"] = path_info[len(script_name) :]

        scheme = environ.get("HTTP_X_FORWARDED_PROTO", "")
        if scheme:
            environ["wsgi.url_scheme"] = scheme
        return wsgi_app(environ, start_response)

    return wrapper


def create_prometheus_metrics(
    project: Optional[str] = None, registry: Optional[CollectorRegistry] = None
) -> GordoServerPrometheusMetrics:
    arg_labels = [("gordo_name", "model")]
    info = {"version": __version__}
    if project is not None:
        info["project"] = project
    else:
        arg_labels.append(("gordo_project", "project"))
    return GordoServerPrometheusMetrics(
        args_labels=arg_labels,
        info=info,
        ignore_paths=["/healthcheck"],
        registry=registry,
    )


def build_app(
    config: Optional[Dict[str, Any]] = None,
    prometheus_registry: Optional[CollectorRegistry] = None,
):
    """
    Build app and any associated routes
    """
    configure_once()

    app = Flask(__name__)
    app.config.from_object(Config())
    if config is not None:
        app.config.update(**config)

    app.register_blueprint(views.base_blueprint)
    app.register_blueprint(views.anomaly_blueprint)

    app.wsgi_app = adapt_proxy_deployment(app.wsgi_app)  # type: ignore
    app.url_map.strict_slashes = False  # /path and /path/ are ok.

    if app.config["ENABLE_PROMETHEUS"]:
        prometheus_metrics = create_prometheus_metrics(
            project=app.config.get("PROJECT"), registry=prometheus_registry
        )
        prometheus_metrics.prepare_app(app)
    elif prometheus_registry is not None:
        logger.warning("Ignoring non empty prometheus_registry argument")

    @app.before_request
    def _start_timer():
        g.start_time = timeit.default_timer()

    @app.before_request
    def _set_revision_and_collection_dir():
        g.collection_dir = os.environ[
            current_app.config["MODEL_COLLECTION_DIR_ENV_VAR"]
        ]
        g.current_revision = os.path.basename(g.collection_dir)

        # If a specific revision was requested, update collection_dir
        g.revision = request.args.get("revision") or request.headers.get("revision")
        if g.revision:
            g.collection_dir = os.path.join(g.collection_dir, "..", g.revision)
            try:
                os.listdir(g.collection_dir)  # List dir to ensure it exists
            except FileNotFoundError:
                return make_response(
                    jsonify({"error": f"Revision '{g.revision}' not found."}), 410
                )
        else:
            g.revision = g.current_revision

    @app.after_request
    def _revision_used(response):
        if response.is_json:
            data = response.get_json()
            data["revision"] = g.revision
            response.set_data(json.dumps(data).encode())
        response.headers["revision"] = g.revision
        return response

    @app.after_request
    def _log_time_taken(response):
        runtime_s = timeit.default_timer() - g.start_time
        logger.debug(f"Total runtime for request: {runtime_s}s")
        response.headers["Server-Timing"] = f"request_walltime_s;dur={runtime_s}"
        return response

    @app.route("/healthcheck")
    def base_healthcheck():
        return "", 200

    @app.route("/server-version")
    def server_version():
        return jsonify({"version": __version__})

    return app


def run_cmd(cmd):
    """
    Run a shell command and handle CalledProcessError and OSError types

    Note
    ----
    This function is abstracted from :func:`~gordo.server.server.run_server`
    in order to test the calling of commands that would allow the subprocess call to
    break, depending on how it is parameterized. For example, calling this without
    sending `stderr` to stdout will cause a segmentation fault when calling an
    executable that does not exist.
    """
    subprocess.check_call(cmd, stderr=subprocess.STDOUT)


def run_server(
    host: str,
    port: int,
    workers: int,
    log_level: str,
    config_module: Optional[str] = None,
    worker_connections: Optional[int] = None,
    threads: Optional[int] = None,
    worker_class: str = "gthread",
    server_app: str = "gordo.server.server:build_app()",
):
    """
    Run application with Gunicorn server using Gevent Async workers

    Parameters
    ----------
    host: str
        The host to run the server on.
    port: int
        The port to run the server on.
    workers: int
        The number of worker processes for handling requests.
    log_level: str
        The log level for the `gunicorn` webserver. Valid log level names can be found
        in the [gunicorn documentation](http://docs.gunicorn.org/en/stable/settings.html#loglevel).
    config_module: str
        The config module. Will be passed with `python:` [prefix](https://docs.gunicorn.org/en/stable/settings.html#config).
    worker_connections: int
        The maximum number of simultaneous clients per worker process.
    threads: str
        The number of worker threads for handling requests.
    worker_class: str
        The type of workers to use.
    server_app: str
        The application to run
    """

    cmd = [
        "gunicorn",
        "--bind",
        f"{host}:{port}",
        "--log-level",
        log_level,
        "--error-logfile",
        "-",
        "--access-logfile",
        "-",
        "--worker-class",
        worker_class,
        "--worker-tmp-dir",
        "/dev/shm",
        "--workers",
        str(workers),
    ]
    if config_module is not None:
        cmd.extend(("--config", "python:" + config_module))
    if worker_class == "gthread":
        if threads is not None:
            cmd.extend(("--threads", str(threads)))
    else:
        if worker_connections is not None:
            cmd.extend(("--worker-connections", str(worker_connections)))

    cmd.append(server_app)
    run_cmd(cmd)


if __name__ == "__main__":
    app = build_app()

    # Run development webserver
    app.run("0.0.0.0", 5555, debug=True, threaded=False)
