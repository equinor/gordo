from flask import Flask
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from prometheus_client.exposition import make_wsgi_app
from .metrics import create_registry


def build_app() -> Flask:
    """
    Creating Flask application in order to serve all Prometheus metrics

    :return: Flask
    """
    curr_app = Flask("gordoserver_prometheus")

    registry = create_registry()

    curr_app.wsgi_app = DispatcherMiddleware(  # type: ignore
        curr_app.wsgi_app, {"/metrics": make_wsgi_app(registry)}
    )

    @curr_app.route("/healthcheck")
    def health_check():
        return "", 200

    return curr_app
