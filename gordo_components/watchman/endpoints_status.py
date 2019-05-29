# -*- coding: utf-8 -*-

import threading
from typing import List, Optional
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor

import requests
from flask import current_app

from gordo_components.watchman.gordo_k8s_interface import (
    Service,
    list_model_builders,
    ModelBuilderPod,
)


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


class EndpointStatuses:
    """
    Represents a thread-safe interface to getting endpoint metadata / statuses
    for use inside of Watchman.

    Can, in a separate thread, call ``EndpointStatuses.update()`` to re-call all
    endpoints and get thier metadata.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._statuses: List[dict] = None

    def update(self):
        """
        Update the object's endpoints to reflect their current state.
        """
        with self._lock:
            self._statuses = self._endpoints_statuses()

    def statuses(self, n_logs: Optional[int] = None) -> List[dict]:
        """
        Return the current List[EndpointStatus] as a list of JSON serializable dicts

        Returns
        -------
        List[dict]
        """
        if not n_logs:
            with self._lock:
                return self._statuses

        # Otherwise we need a 'fresh' view of the statuses to include latest logs
        else:
            return self._endpoints_statuses(n_logs)

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
            Name of the target, aka machine-name
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
    def _endpoints_statuses(n_logs: Optional[int] = None) -> List[dict]:

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
                    EndpointStatuses._check_endpoint,
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
