# -*- coding: utf-8 -*-
import logging
from collections import namedtuple
from datetime import datetime
from typing import List, Callable

import apscheduler.schedulers.base
import kubernetes
import pytz
import requests

from gordo_components.watchman.gordo_k8s_interface import watch_service

logger = logging.getLogger(__name__)

EndpointStatus = namedtuple(
    "EndpointStatus",
    ["endpoint", "target", "endpoint_metadata", "healthy", "last_seen"],
)


def endpoint_url_for_model(project_name: str, model_name: str) -> str:
    """Calculates the endpoint URL for a model"""
    return f"gordo/v0/{project_name}/{model_name}/"


def rename_underscored_keys_to_dashes(d: dict) -> dict:
    """Renames keys like `endpoint_metadata` to `endpoint-metadata`
    inplace in the passed dictionary"""
    for k in d.copy().keys():
        if "_" in k:
            d[k.replace("_", "-")] = d.pop(k)
    return d


def job_name_for_target_update(target_name: str) -> str:
    """ The name of the update model metadata APscheduler job for a target"""
    return f"update_model_metadata_{target_name}"


class EndpointStatuses:
    """
    Represents a interface to getting endpoint metadata / statuses for use
    inside of Watchman. If `listen_to_kubernetes` is true (default and
    recommended) then it will listen to updated from kubernetes, and fetch
    updated metadata when a change is detected. If it is false then we will
    try to retrieve updates every 10 seconds until a success, and then every 5
    minutes after a successful fetch.

    """

    def __init__(
        self,
        scheduler: apscheduler.schedulers.base,
        project_name: str,
        ambassador_host: str,
        target_names: List[str],
        project_version: str = None,
        namespace: str = None,
        listen_to_kubernetes=True,
    ):
        """

        Parameters
        ----------
        scheduler: apscheduler.schedulers.base
            Scheduler to be used for running misc jobs
        project_name: str
            Project name
        ambassador_host:
            Full hostname of ambassador
        target_names: List[str]
            List of all target names.
        namespace: str
            Namespace to listen for new services in. Irrelevant if
            `listen_to_kubernetes`==False
        project_version: str
            Project version to listen for updates to. Irrelevant if
            `listen_to_kubernetes`==False
        listen_to_kubernetes: bool
            If true then listen for updates from kubernetes (recommended),
            otherwise just try to fetch metadata from the targets
            at regular intervals.

        """
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
            # If we are listening to kubernetes events
            # we only do a "manual" update of every endpoint every 20 min
            self.update_model_interval = 1200
            watcher = watch_for_model_server_service(
                namespace=namespace,
                project_name=self.project_name,
                project_version=project_version,
                processor=self.handle_k8s_model_service_event,
            )

            watcher.start()
        else:
            # If we are not listening to kubernetes events
            # we do a "manual" update of every endpoint every 5 min
            self.update_model_interval = 300
            for target in target_names:
                self._schedule_update_for_target(target, seconds=10)
                self.update_model_metadata(target)

    def statuses(self,) -> List[dict]:
        """
        Return the current List[EndpointStatus] as a list of JSON
        serializable dicts.

        Returns
        -------
        List[dict]
        """
        return [
            rename_underscored_keys_to_dashes(es._asdict())
            for es in self.model_metadata.values()
        ]

    def handle_k8s_model_service_event(
        self, event: kubernetes.client.models.v1_watch_event.V1WatchEvent
    ):
        """
        Handles a kubernetes model service update event. Either schedules
        a update-job, or removes it if the event is `DELETE`


        Parameters
        ----------
        event: kubernetes.client.models.v1_watch_event.V1WatchEvent
            Event from kubernetes

        Returns
        -------

        """
        target_name = event.object.metadata.labels.get(
            "applications.gordo.equinor.com/model-name", None
        )
        event_type = event.type
        logger.info(f"Got K8s event for model {target_name} of type {event_type}")

        if event_type == "DELETED":
            job_name = job_name_for_target_update(target_name)
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
                    "Got updated model-server service notification, but found "
                    "no model-name"
                )

    def _schedule_update_for_target(self, target_name, seconds=2):
        """Schedules a regular update job for the target, every `seconds`
        seconds. If the job exists it will be updated with the new schedule,
        and if it does not exist it will be added with next_run_time = now
        and interval = `seconds`.

        """
        job_name = job_name_for_target_update(target_name)
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

    def update_model_metadata(self, target_name: str):
        """
        Updates model metadata for a target, and if it succeeds then reschedule the
        update job to happen every `self.update_model_interval` second.

        Parameters
        ----------
        target_name: str
            Name of target to update metadata for.

        """
        logger.info(f"Checking target {target_name}")
        endpoint = fetch_single_target_metadata(
            host=self.host,
            target=target_name,
            endpoint=endpoint_url_for_model(self.project_name, target_name),
        )
        if endpoint.healthy:
            logger.info(
                f"Found healthy target {target_name}, "
                f"saving metadata and rescheduling update job"
            )
            self.model_metadata[target_name] = endpoint
            self._schedule_update_for_target(
                target_name=target_name, seconds=self.update_model_interval
            )
        else:
            logger.info(f"Found that target {target_name} was not up yet")


def fetch_single_target_metadata(
    host: str, target: str, endpoint: str
) -> EndpointStatus:
    """
    Check if a given endpoint returning metadata about it.

    Parameters
    ----------
    host: str
        Name of the host to query, e.g. `ambassador`
    target: str
        Name of the target, aka model-name
    endpoint: str
        Relative (to host) url of the endpoint, e.g. `/gordo/v0/model1`

    Returns
    -------
    EndpointStatus
    """
    endpoint = endpoint.lstrip("/").rstrip("/")
    base_url = f"http://{host}/{endpoint}"
    metadata: dict = dict()
    metadata_resp_ok = False
    try:
        metadata_url = f"{base_url}/metadata"
        logger.info(f"Trying to fetch metadata from url {metadata_url}")
        metadata_resp = requests.get(metadata_url, timeout=2)
        logger.info(f"Url {metadata_url} gave exit code: {metadata_resp.status_code}")
        metadata_resp_ok = metadata_resp.ok
        metadata = metadata_resp.json()
    except requests.exceptions.RequestException:
        logger.info(f"Failed getting metadata for endpoint {base_url}")
    return EndpointStatus(
        endpoint="/" + endpoint,
        target=target,
        endpoint_metadata=metadata,
        healthy=metadata_resp_ok,
        last_seen=datetime.now(pytz.utc).isoformat(),
    )


def watch_for_model_server_service(
    namespace: str, project_name: str, project_version: str, processor: Callable
):
    selectors = {
        "applications.gordo.equinor.com/project-name": project_name,
        "applications.gordo.equinor.com/project-version": project_version,
        "app.kubernetes.io/component": "service",
        "app.kubernetes.io/managed-by": "gordo",
        "app.kubernetes.io/name": "model-server",
        "app.kubernetes.io/part-of": "gordo",
    }

    return watch_service(processor=processor, namespace=namespace, selectors=selectors)
