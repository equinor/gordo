# -*- coding: utf-8 -*-
import logging
from datetime import datetime
from typing import Iterable
from collections import namedtuple

import apscheduler.schedulers.base
import kubernetes
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


def rename_underscored_keys_to_dashes(d: dict):
    """Renames keys like `endpoint_metadata` to `endpoint-metadata`"""
    for k in d.copy().keys():
        if "_" in k:
            d[k.replace("_", "-")] = d.pop(k)
    return d


def model_metadata_to_response_list(model_metadata):
    return [
        # Renames endpoint_metadata and last_seen to dashed versions
        rename_underscored_keys_to_dashes(es._asdict())
        for es in model_metadata.values()
    ]


def job_name_for_target(target_name: str) -> str:
    job_name = f"update_model_metadata_{target_name}"
    return job_name


class EndpointStatuses:
    """
    Represents a interface to getting endpoint metadata / statuses for use inside
    of Watchman.

    """

    def __init__(
        self,
        scheduler: apscheduler.schedulers.base,
        project_name: str,
        namespace: str,
        project_version: str,
        ambassador_host: str,
        target_names: Iterable[str],
        listen_to_kubernetes=True,
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
        self.host = ambassador_host
        self.scheduler = scheduler
        if listen_to_kubernetes:
            watcher = watch_for_model_server_service(
                namespace=namespace,
                project_name=self.project_name,
                project_version=project_version,
                processor=self.handle_updated_model_service_event,
            )

            watcher.start()
        else:
            for target in target_names:
                self.update_model_metadata(target)

    def statuses(self,) -> Iterable[dict]:
        """
        Return the current Iterable[EndpointStatus] as a list of JSON serializable dicts

        Returns
        -------
        List[dict]
        """

        return model_metadata_to_response_list(self.model_metadata)

    def handle_updated_model_service_event(
        self, event: kubernetes.client.models.v1_watch_event.V1WatchEvent
    ):
        target_name = event.object.metadata.labels.get(
            "applications.gordo.equinor.com/model-name", None
        )
        event_type = event.type
        logger.info(f"Got K8s event for model {target_name} of type {event_type}")

        if event_type == "DELETED":
            job_name = job_name_for_target(target_name)
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
                self._schedule_update_for_target(target_name)
            else:
                logger.warning(
                    "Got updated model-server service notification, but found no "
                    "model-name"
                )

    def _schedule_update_for_target(self, target_name, seconds=2):
        """Schedules a regular update job for the target, every `seconds`
        seconds. If the job exists it will be updated with the new schedule,
        and if it does not exist it will be added with next_run_time = now
        and interval = `seconds`.

        """
        job_name = job_name_for_target(target_name)
        logger.info(f"Adding update model metadata job: {job_name}")
        if self.scheduler.get_job(job_name):
            self.scheduler.reschedule_job(
                job_id=job_name, trigger="interval", seconds=seconds
            )
        else:
            self.scheduler.add_job(
                func=self.update_model_metadata,
                trigger="interval",
                seconds=seconds,
                coalesce=True,
                max_instances=1,
                id=job_name,
                next_run_time=datetime.now(),
                kwargs={"target_name": target_name},
            )

    def update_model_metadata(self, target_name):
        logger.info(f"Checking target {target_name}")
        endpoint = _check_target(
            host=self.host, target=target_name, project_name=self.project_name
        )
        if endpoint.healthy:
            logger.info(
                f"Found healthy target {target_name}, "
                f"saving metadata and rescheduling update job"
            )
            self.model_metadata[target_name] = endpoint
            self._schedule_update_for_target(target_name=target_name, seconds=300)
        else:
            logger.info(f"Found that target {target_name} was not up yet")


def _check_target(host: str, target: str, project_name: str) -> EndpointStatus:
    """
    Check if a given endpoint returning metadata about it.

    Parameters
    ----------
    host: str
        Name of the host to query
    target: str
        Name of the target, aka model-name
    Returns
    -------
    EndpointStatus
    """
    endpoint = endpoint_url_for_model(project_name, target).lstrip("/").rstrip("/")
    base_url = f"http://{host}/{endpoint}"
    try:
        metadata_url = f"{base_url}/metadata"
        logger.info(f"Trying to fetch metadata from url {metadata_url}")
        metadata_resp = requests.get(metadata_url, timeout=2)
        logger.info(f"Url {metadata_url} gave exit code: {metadata_resp.status_code}")
    except requests.exceptions.RequestException:
        metadata_resp = None
    metadata_resp_ok = metadata_resp.ok if metadata_resp else False
    metadata = metadata_resp.json() if metadata_resp_ok else dict()

    return EndpointStatus(
        endpoint="/" + endpoint,
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
