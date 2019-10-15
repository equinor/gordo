# -*- coding: utf-8 -*-
"""
This module contains code for generating the Gordo server Flask application.

Running this module will run the application using Flask's development webserver.
Gunicorn can be used to run the application as `gevent` async workers by using the
:func:`~gordo_components.server.server.run_server` function.
"""
import logging
import timeit
import typing
from functools import wraps
import subprocess

from flask import Flask, g
from gordo_components.server import views

logger = logging.getLogger(__name__)


class Config:
    """Server config"""

    MODEL_COLLECTION_DIR_ENV_VAR = "MODEL_COLLECTION_DIR"


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


def build_app():
    """
    Build app and any associated routes
    """
    app = Flask(__name__)
    app.config.from_object(Config())

    app.register_blueprint(views.base_blueprint)
    app.register_blueprint(views.anomaly_blueprint)

    app.wsgi_app = adapt_proxy_deployment(app.wsgi_app)  # type: ignore
    app.url_map.strict_slashes = False  # /path and /path/ are ok.

    @app.before_request
    def _start_timer():
        g.start_time = timeit.default_timer()

    @app.after_request
    def _log_time_taken(response):
        runtime_s = timeit.default_timer() - g.start_time
        logger.debug(f"Total runtime for request: {runtime_s}s")
        response.headers["Server-Timing"] = f"request_walltime_s;dur={runtime_s}"
        return response

    @app.route("/healthcheck")
    def base_healthcheck():
        return "", 200

    return app


def run_cmd(cmd):
    """
    Run a shell command and handle CalledProcessError and OSError types

    Note
    ----
    This function is abstracted from :func:`~gordo_components.server.server.run_server`
    in order to test the calling of commands that would allow the subprocess call to
    break, depending on how it is parameterized. For example, calling this without
    sending `stderr` to stdout will cause a segmentation fault when calling an
    executable that does not exist.
    """
    subprocess.check_call(cmd, stderr=subprocess.STDOUT)


def run_server(
    host: str, port: int, workers: int, worker_connections: int, log_level: str
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
    worker_connections: int
        The maximum number of simultaneous clients per worker process.
    log_level: str
        The log level for the `gunicorn` webserver. Valid log level names can be found
        in the [gunicorn documentation](http://docs.gunicorn.org/en/stable/settings.html#loglevel).
    """

    cmd = [
        "gunicorn",
        "--bind",
        f"{host}:{port}",
        "--log-level",
        log_level,
        "--worker-class",
        "gevent",
        "--worker-tmp-dir",
        "/dev/shm",
        "--workers",
        str(workers),
        "--worker-connections",
        str(worker_connections),
        "gordo_components.server.server:app",
    ]
    run_cmd(cmd)


app = build_app()

if __name__ == "__main__":
    # Run development webserver
    app.run("0.0.0.0", 5555, debug=True, threaded=False)
