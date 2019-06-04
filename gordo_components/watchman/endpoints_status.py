# -*- coding: utf-8 -*-
import logging
from datetime import datetime
from typing import List, Iterable
from collections import namedtuple

import apscheduler.schedulers.base
import requests
import pytz


from gordo_components.watchman.gordo_k8s_interface import watch_service


logger = logging.getLogger(__name__)

EndpointStatus = namedtuple(
    "EndpointStatus",
    ["endpoint", "target", "endpoint_metadata", "healthy", "last_seen"],
)


def endpoint_url_for_model(project_name, model_name):
    endpoint_url = f"/gordo/v0/{project_name}/{model_name}/"
    return endpoint_url


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
        target_names: Iterable[str],
    ):
        self.model_metadata: dict = {
            target_name: EndpointStatus(
                endpoint=endpoint_url_for_model(project_name, target_name),
                target=target_name,
                healthy=False,
                endpoint_metadata=dict(),
                last_seen=None,
            )
            for target_name in target_names
        }
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

    def statuses(self) -> Iterable[dict]:
        """
        Return the current Iterable[EndpointStatus] as a list of JSON serializable dicts

        Returns
        -------
        List[dict]
        """

        return (es._asdict() for es in self.model_metadata.values())

    def handle_updated_model_service_event(self, event):
        target_name = event["object"].metadata.labels.get(
            "applications.gordo.equinor.com/machine-name", None
        )
        event_type = event.get("type", None)
        logger.info(f"Got K8s event for model {target_name} of type {event_type}")
        job_name = f"update_model_metadata_{target_name}"
        if event_type == "DELETED":
            if target_name and self.scheduler.get_job(job_name):
                logger.info(f"Removing job {job_name}")
                if target_name in self.model_metadata:
                    self.model_metadata[target_name] = self.model_metadata[
                        target_name
                    ]._replace(
                        healthy=False, last_seen=datetime.now(pytz.utc).isoformat()
                    )
                self.scheduler.remove_job(job_id=job_name)
        else:
            if target_name:
                logger.info(f"Adding update model metadata job: {job_name}")
                self.scheduler.add_job(
                    func=self.update_model_metadata,
                    trigger="interval",
                    seconds=2,
                    coalesce=True,
                    max_instances=1,
                    id=job_name,
                    next_run_time=datetime.now(),
                    kwargs={"target_name": target_name},
                )
            else:
                logger.warning(
                    "Got updated model-server service notification, but found no machine-name"
                )

    def update_model_metadata(self, target_name):
        endpoint_url = endpoint_url_for_model(self.project_name, target_name)
        logger.info(f"Checking endpoint {endpoint_url}")
        endpoint = _check_endpoint(
            host=self.host, target=target_name, endpoint=endpoint_url
        )
        if endpoint.healthy:
            logger.info(
                f"Found healthy endpoint {endpoint_url}, saving metadata and rescheduling update job"
            )
            job_name = f"update_model_metadata_{target_name}"
            self.model_metadata[target_name] = endpoint
            self.scheduler.reschedule_job(
                job_id=job_name, trigger="interval", minutes=5
            )
        else:
            logger.info(f"Found that endpoint {endpoint_url} was not up yet")


def _check_endpoint(host: str, target: str, endpoint: str) -> EndpointStatus:
    """
    Check if a given endpoint returning metadata about it.

    Parameters
    ----------
    host: str
        Name of the host to query
    target: str
        Name of the target, aka machine-name
    endpoint: str
        Endpoint to check. ie. /gordo/v0/test-project/test-machine
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

    return EndpointStatus(
        endpoint=endpoint,
        target=target,
        endpoint_metadata=metadata,
        healthy=metadata_resp_ok,
        last_seen=datetime.now(pytz.utc).isoformat(),
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
