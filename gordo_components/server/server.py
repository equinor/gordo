# -*- coding: utf-8 -*-
import os
import logging
import timeit
import typing
from functools import wraps

from flask import Flask, g
from sklearn.base import BaseEstimator

from gordo_components import serializer
from gordo_components.data_provider.base import GordoBaseDataProvider
from gordo_components.server import views

logger = logging.getLogger(__name__)


class Config:
    """Server config"""

    MODEL_LOCATION_ENV_VAR = "MODEL_LOCATION"


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


def load_model_and_metadata(
    model_dir_env_var: str
) -> typing.Tuple[BaseEstimator, dict]:
    """
    Loads a model and metadata from the path found in ``model_dir_env_var``
    environment variable

    Parameters
    ----------
    model_dir_env_var: str
        The name of the environment variable which stores the location of the model

    Returns
    -------
    BaseEstimator, dict
        Tuple where the 0th element is the model, and the 1st element is the metadata
        associated with the model
    """
    logger.debug("Determining model location...")
    model_location = os.getenv(model_dir_env_var)
    if model_location is None:
        raise ValueError(f'Environment variable "{model_dir_env_var}" not set!')
    if not os.path.isdir(model_location):
        raise NotADirectoryError(
            f'The supplied directory: "{model_location}" does not exist!'
        )

    model = serializer.load(model_location)
    metadata = serializer.load_metadata(model_location)
    return model, metadata


def build_app(data_provider: typing.Optional[GordoBaseDataProvider] = None):
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
    def _reg_data_provider():
        g.data_provider = data_provider

    @app.before_request
    def _start_timer():
        g.start_time = timeit.default_timer()

    @app.after_request
    def _log_time_taken(response):
        runtime_s = timeit.default_timer() - g.start_time
        logger.debug(f"Total runtime for request: {runtime_s}s")
        response.headers["Server-Timing"] = f"request_walltime_s;dur={runtime_s}"
        return response

    with app.app_context():
        app.model, app.metadata = load_model_and_metadata(
            app.config["MODEL_LOCATION_ENV_VAR"]
        )

    return app


def run_server(
    host: str = "0.0.0.0",
    port: int = 5555,
    debug: bool = False,
    data_provider: typing.Optional[GordoBaseDataProvider] = None,
):
    app = build_app(data_provider=data_provider)
    app.run(host, port, debug=debug)


if __name__ == "__main__":
    run_server()
