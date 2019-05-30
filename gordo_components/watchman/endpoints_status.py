# -*- coding: utf-8 -*-
import threading
import logging
from datetime import datetime
from typing import List, Optional
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor

import apscheduler.schedulers.base
import requests
from flask import current_app

from gordo_components.watchman.gordo_k8s_interface import (
    Service,
    list_model_builders,
    ModelBuilderPod,
    watch_service,
)


logger = logging.getLogger(__name__)

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

    """

    def __init__(
        self,
        scheduler: apscheduler.schedulers.base,
        project_name: str,
        namespace: str,
        project_version: str,
        ambassador_namespace: str,
    ):
        self._lock = threading.Lock()
        self._statuses: List[dict] = None
        self.model_metadata: dict = {}
        self.project_name = project_name
        self.namespace = namespace
        self.project_version = project_version
        self.host = f"ambassador.{ambassador_namespace}"
        # self.host = f"localhost:8000"

        self.scheduler = scheduler
        watcher = watch_for_model_server_service(
            namespace=self.namespace,
            project_name=self.project_name,
            project_version=self.project_version,
            processor=self.handle_updated_model_service_event,
        )

        watcher.start()

    def statuses(self, n_logs: Optional[int] = None) -> List[dict]:
        """
        Return the current List[EndpointStatus] as a list of JSON serializable dicts

        Returns
        -------
        List[dict]
        """
        if not n_logs:
            ret = []
            for endpoint, metadata in self.model_metadata.items():
                ret.append(
                    {
                        "endpoint": endpoint,
                        "healthy": True,
                        "endpoint-metadata": metadata,
                    }
                )
            return ret

        # Otherwise we need a 'fresh' view of the statuses to include latest logs
        else:
            return self._endpoints_statuses(n_logs)

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
                    _check_endpoint,
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

    def handle_updated_model_service_event(self, event):
        model_name = event["object"].metadata.labels.get(
            "applications.gordo.equinor.com/machine-name", None
        )
        event_type = event.get("type", None)
        logger.info(f"Got K8s event for model {model_name} of type {event_type}")
        job_name = f"update_model_metadata_{model_name}"
        if event_type == "DELETED":
            if model_name and self.scheduler.get_job(job_name):
                logger.info(f"Removing job {job_name}")
                del self.model_metadata[model_name]
                self.scheduler.remove_job(job_id=job_name)
        else:
            if model_name:
                logger.info(f"Adding update model metadata job: {job_name}")
                self.scheduler.add_job(
                    func=self.update_model_metadata,
                    trigger="interval",
                    seconds=2,
                    coalesce=True,
                    max_instances=1,
                    id=job_name,
                    next_run_time=datetime.now(),
                    kwargs={"model_name": model_name},
                )
            else:
                logger.warning(
                    "Got updated model-server service notification, but found no machine-name"
                )

    def update_model_metadata(self, model_name):
        endpoint_url = self.endpoint_url_for_model(model_name)
        logger.info(f"Checking endpoint {endpoint_url}")
        endpoint = _check_endpoint(
            host=self.host,
            namespace=self.namespace,
            project_name=self.project_name,
            target=model_name,
            endpoint=endpoint_url,
            builder_pod=None,
            n_logs=None,
        )
        if endpoint.healthy:
            logger.info(
                f"Found healthy endpoint {endpoint_url}, saving metadata and rescheduling update job"
            )
            job_name = f"update_model_metadata_{model_name}"
            self.model_metadata[model_name] = endpoint
            self.scheduler.reschedule_job(
                job_id=job_name, trigger="interval", minutes=5
            )
        else:
            logger.info(f"Found that endpoint {endpoint_url} was not up yet")

    def endpoint_url_for_model(self, model_name):
        endpoint_url = f"/gordo/v0/{self.project_name}/{model_name}/"
        return endpoint_url


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
    try:
        metadata_resp = requests.get(f"{base_url}/metadata", timeout=2)
    except requests.exceptions.RequestException:
        metadata_resp = None
    metadata_resp_ok = metadata_resp.ok if metadata_resp else False
    metadata = metadata_resp.json() if metadata_resp_ok else dict()

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
        healthy=metadata_resp_ok,
        model_builder_status=ModelBuilderStatus(
            status=status_model_builder, logs=logs_model_builder
        ),
        model_server_status=ModelServerStatus(status=service_status, logs=service_logs),
    )


def watch_for_model_server_service(namespace, project_name, project_version, processor):
    selectors = {
        "applications.gordo.equinor.com/project-name": project_name,
        "applications.gordo.equinor.com/project-version": project_version,
        "app.kubernetes.io/component": "service",
        "app.kubernetes.io/managed-by": "gordo",
        "app.kubernetes.io/name": "model-server",
        "app.kubernetes.io/part-of": "gordo",
    }

    return watch_service(processor=processor, namespace=namespace, selectors=selectors)
