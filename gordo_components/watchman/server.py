# -*- coding: utf-8 -*-

import os
import logging
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, Optional, List

import requests
from cachetools import cached, TTLCache
from flask import Flask, jsonify, make_response, request, current_app
from flask.views import MethodView

from gordo_components import __version__
from gordo_components.watchman.gordo_k8s_interface import (
    Service,
    list_model_builders,
    ModelBuilderPod,
)


logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("WATCHMAN_LOGLEVEL", "INFO").upper())


EndpointStatus = namedtuple(
    "EndpointStatus",
    [
        "endpoint",
        "target",
        "metadata",
        "healthy",
        "model_builder_status",
        "model_server_status",
    ],
)
ModelBuilderStatus = namedtuple("ModelBuilderStatus", "status logs")
ModelServerStatus = namedtuple("ModelServerStatus", "status logs")


class WatchmanApi(MethodView):
    """
    API view to list expected endpoints in this project space and report if they
    are up or not.
    """

    @staticmethod
    def _check_endpoint(
        host: str,
        namespace: str,
        project_name: str,
        target: str,
        endpoint: str,
        builder_pod: Optional[ModelBuilderPod] = None,
        n_logs: Optional[int] = None,
    ) -> EndpointStatus:
        """
        Check if a given endpoint returning metadata about it.

        Parameters
        ----------
        host: str
            Name of the host to query
        namespace: str
            Namespace k8s client should operate in.
        project_name: str
            The name of this project we're going to query for.
        target: str
            Name of the target, aka model name
        endpoint: str
            Endpoint to check. ie. /gordo/v0/test-project/test-machine
        builder_pod: Optional[ModelBuilderPod]
            A reference to the pod responsible for building the model for this target
        n_logs: int
            Number of lines worth of logs to fetch from builders and services

        Returns
        -------
        EndpointStatus
        """

        endpoint = endpoint[1:] if endpoint.startswith("/") else endpoint
        base_url = f'http://{host}/{endpoint.rstrip("/")}'

        metadata_resp = requests.get(f"{base_url}/metadata", timeout=2)
        metadata = metadata_resp.json() if metadata_resp.ok else dict()

        # Get model builder status / logs
        if builder_pod is not None:
            # Get the current phase and logs of the pod responsible for model building
            status_model_builder, logs_model_builder = (
                builder_pod.status.phase,
                builder_pod.logs() if n_logs is None else builder_pod.logs(n_logs),
            )
        else:
            status_model_builder, logs_model_builder = None, None  # type: ignore

        # Get server (a Kubernetes service) status / logs
        if n_logs is not None:
            service = Service(  # type: ignore
                namespace=namespace, name=f"gordoserver-{project_name}-{target}"
            )
            service_status, service_logs = service.status, service.logs(n_logs)
        else:
            service_status, service_logs = None, None  # type: ignore

        return EndpointStatus(
            endpoint=endpoint,
            target=target,
            metadata=metadata,
            healthy=metadata_resp.ok,
            model_builder_status=ModelBuilderStatus(
                status=status_model_builder, logs=logs_model_builder
            ),
            model_server_status=ModelServerStatus(
                status=service_status, logs=service_logs
            ),
        )

    @staticmethod
    @cached(cache=TTLCache(maxsize=1024, ttl=5))
    def _endpoints_statuses(n_logs: Optional[int]) -> List[dict]:

        # Get a list of ModelBuilderPod instances and map to target names
        if n_logs is not None:
            builders = list_model_builders(
                namespace=current_app.config["NAMESPACE"],
                project_name=current_app.config["PROJECT_NAME"],
                project_version=current_app.config["PROJECT_VERSION"],
            )
            builders = {pod.target_name: pod for pod in builders}
        else:
            builders = {}

        with ThreadPoolExecutor(max_workers=25) as executor:
            futures = {
                executor.submit(
                    WatchmanApi._check_endpoint,
                    host=f'ambassador.{current_app.config["AMBASSADOR_NAMESPACE"]}',
                    namespace=current_app.config["NAMESPACE"],
                    project_name=current_app.config["PROJECT_NAME"],
                    target=target,
                    endpoint=endpoint,
                    builder_pod=builders.get(target),
                    n_logs=n_logs,
                ): endpoint
                for target, endpoint in zip(
                    current_app.config["TARGET_NAMES"], current_app.config["ENDPOINTS"]
                )
            }

            # List of dicts: [{'endpoint': /path/to/endpoint, 'healthy': bool, 'metadata': dict, ...}]
            status_results = []
            for f in futures:

                exception = f.exception()

                if exception is not None:
                    status_results.append(
                        {
                            "endpoint": futures[f],
                            "healthy": False,
                            "error": f"Unable to properly probe endpoint: {exception}",
                        }
                    )
                else:
                    status = f.result()  # type: EndpointStatus

                    status_results.append(
                        {
                            "endpoint": futures[f],
                            "healthy": status.healthy,
                            "endpoint-metadata": status.metadata,
                            "model-server": {
                                "logs": status.model_server_status.logs,
                                "status": status.model_server_status.status,
                            },
                            "model-builder": {
                                "logs": status.model_builder_status.logs,
                                "status": status.model_builder_status.status,
                            },
                        }
                    )
            return status_results

    def get(self):

        n_logs = int(request.args.get("logs") or 20) if "logs" in request.args else None
        status_results = self._endpoints_statuses(n_logs)

        payload = jsonify(
            {
                "endpoints": status_results,
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
