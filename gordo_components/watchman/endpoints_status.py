# -*- coding: utf-8 -*-
import logging
from collections import namedtuple
from datetime import datetime
from typing import List, Callable, Optional

import apscheduler.schedulers.base
import kubernetes
import pytz
import requests

from gordo_components.watchman.gordo_k8s_interface import watch_namespaced_services

logger = logging.getLogger(__name__)

EndpointStatus = namedtuple(
    "EndpointStatus",
    ["endpoint", "target", "endpoint_metadata", "healthy", "last_checked", "last_seen"],
)


def endpoint_url_for_model(project_name: str, model_name: str) -> str:
    """Calculates the endpoint URL for a model"""
    return f"gordo/v0/{project_name}/{model_name}/"


def job_name_for_model_update(model_name: str) -> str:
    """ The name of the update model metadata APscheduler job for a model"""
    return f"update_model_metadata_{model_name}"


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
        model_names: List[str],
        project_version: Optional[str] = None,
        namespace: Optional[str] = None,
        listen_to_kubernetes=True,
    ):
        """

        Parameters
        ----------
        scheduler: apscheduler.schedulers.base
            Scheduler to be used for running misc jobs
        project_name: str
            Project name
        ambassador_host: str
            Full hostname of ambassador
        model_names: List[str]
            List of all model names.
        namespace: Optional[str]
            Namespace to listen for new services in. Required if
            `listen_to_kubernetes`==True
        project_version: Optional[str]
            Project version to listen for updates to. Required if
            `listen_to_kubernetes`==True
        listen_to_kubernetes: bool
            If true then listen for updates from kubernetes (recommended),
            otherwise just try to fetch metadata from the models
            at regular intervals.

        """
        self.model_metadata: dict = {
            model_name: EndpointStatus(
                endpoint=endpoint_url_for_model(project_name, model_name),
                target=model_name,
                healthy=False,
                endpoint_metadata=dict(),
                last_checked=None,
                last_seen=None,
            )
            for model_name in model_names
        }
        self.project_name = project_name
        self.host = ambassador_host
        self.scheduler = scheduler
        if listen_to_kubernetes:
            if namespace is None or project_version is None:
                raise ValueError(
                    "Namespace and project_version required when "
                    "`listen_to_kubernetes`==True"
                )
            # If we are listening to kubernetes events
            # we only do a "manual" update of every endpoint every 20 min
            self.happy_update_model_interval = 1200
            watcher = watch_for_model_server_service(
                namespace=namespace,
                project_name=self.project_name,
                project_version=project_version,
                event_handler=self._handle_k8s_model_service_event,
            )

            watcher.start()
        else:
            # If we are not listening to kubernetes events
            # we do a "manual" update of every endpoint every 5 min
            self.happy_update_model_interval = 300
            for model in model_names:
                self._schedule_update_for_model(model, seconds=10)
                self.update_model_metadata(model)

    def statuses(self,) -> List[dict]:
        """
        Return the current List[EndpointStatus] as a list of JSON
        serializable dicts.

        Returns
        -------
        List[dict]
        """

        return [
            {k.replace("_", "-"): v for k, v in es._asdict().items()}
            for es in self.model_metadata.values()
        ]

    def _handle_k8s_model_service_event(self, event: dict):
        """
        Handles a kubernetes model service update event. Either schedules
        an update-job, or removes it if the event is `DELETE`.


        Parameters
        ----------
        event: dict
            Event from kubernetes, see `kubernetes.watch.watch.Watch.stream` for
            description of intended event structure. Needs to have at least a key
            `object`and a key `type`. 'object' should be of type
            `kubernetes.client.models.v1_service.V1Service`

        Returns
        -------
        None
        """
        event_obj: kubernetes.client.models.v1_service.V1Service = event["object"]
        logger.debug(f"Full k8s event: {event}")
        model_name = None
        if event_obj.metadata is not None and event_obj.metadata.labels is not None:
            model_name = event_obj.metadata.labels.get(
                "applications.gordo.equinor.com/model-name", None
            )

        event_type = event["type"]
        logger.info(f"Got K8s event for model {model_name} of type {event_type}")

        if model_name:
            if event_type == "DELETED":
                job_name = job_name_for_model_update(model_name)
                if self.scheduler.get_job(job_name):
                    logger.info(f"Removing job {job_name}")
                    self._unhealty_model(model_name)
                    self.scheduler.remove_job(job_id=job_name)
                else:
                    logger.warning(
                        f"Retrieved a delete event for model {model_name} but had no "
                        f"matching job "
                    )
            else:
                self._schedule_update_for_model(model_name, seconds=2)
        else:
            logger.warning(
                f"Got model-server service notification, but found no model-name. "
                f"Full event: {event} "
            )

    def _unhealty_model(self, model_name):
        """Registers that a model is no longer available"""
        if model_name in self.model_metadata:
            old_model_metadata = self.model_metadata[model_name]
            new_model_metadata = old_model_metadata._replace(
                healthy=False, last_checked=datetime.now(pytz.utc).isoformat()
            )
            self.model_metadata[model_name] = new_model_metadata

    def _schedule_update_for_model(self, model_name, seconds=5):
        """Schedules a regular model-update job for the model, every `seconds`
        seconds. If the job exists it will be updated with the new schedule,
        and if it does not exist it will be added with next_run_time = now
        and interval = `seconds`.

        """
        job_name = job_name_for_model_update(model_name)
        if self.scheduler.get_job(job_name):
            logger.info(
                f"Updating scheduled job {job_name} to run every {seconds} seconds"
            )
            self.scheduler.reschedule_job(
                job_id=job_name, trigger="interval", seconds=seconds
            )
        else:
            logger.info(
                f"Adding scheduled job {job_name} to run every {seconds} seconds"
            )
            self.scheduler.add_job(
                func=self.update_model_metadata,
                trigger="interval",
                seconds=seconds,
                coalesce=True,
                max_instances=1,
                id=job_name,
                next_run_time=datetime.now(),
                kwargs={"model_name": model_name},
            )

    def update_model_metadata(self, model_name: str):
        """
        Updates model metadata for a model, and if it succeeds then reschedule the
        update job to happen every `self.update_model_interval` second. If it fails then
        ensure that we are scheduled to update frequently.

        Parameters
        ----------
        model_name: str
            Name of model to update metadata for.

        """
        logger.info(f"Checking model {model_name}")
        endpoint = fetch_single_model_metadata(
            host=self.host,
            model_name=model_name,
            endpoint_url=endpoint_url_for_model(self.project_name, model_name),
        )
        if endpoint.healthy:
            logger.info(
                f"Found healthy model {model_name}, "
                f"saving metadata and rescheduling update job"
            )
            self.model_metadata[model_name] = endpoint
            self._schedule_update_for_model(
                model_name=model_name, seconds=self.happy_update_model_interval
            )
        else:
            logger.info(f"Found that model {model_name} was not up yet")
            self._unhealty_model(model_name)


def fetch_single_model_metadata(
    host: str, model_name: str, endpoint_url: str
) -> EndpointStatus:
    """
    Fetch metadata for a single model

    Parameters
    ----------
    host: str
        Name of the host to query, e.g. `ambassador`
    model_name: str
        Name of the model
    endpoint_url: str
        Relative (to host) url of the endpoint, e.g. `/gordo/v0/model1`

    Returns
    -------
    EndpointStatus
    """
    endpoint_url = endpoint_url.lstrip("/").rstrip("/")
    base_url = f"http://{host}/{endpoint_url}"
    metadata: dict = dict()
    metadata_resp_ok = False
    try:
        metadata_url = f"{base_url}/metadata"
        logger.info(f"Trying to fetch metadata from url {metadata_url}")
        metadata_resp = requests.get(metadata_url, timeout=2)
        logger.info(f"Url {metadata_url} gave exit code: {metadata_resp.status_code}")
        metadata_resp_ok = metadata_resp.ok
        if metadata_resp_ok:
            metadata = metadata_resp.json()
    except requests.exceptions.RequestException:
        logger.info(f"Failed getting metadata for endpoint {base_url}")
    return EndpointStatus(
        endpoint="/" + endpoint_url,
        target=model_name,
        endpoint_metadata=metadata,
        healthy=metadata_resp_ok,
        last_checked=datetime.now(pytz.utc).isoformat(),
        last_seen=datetime.now(pytz.utc).isoformat() if metadata_resp_ok else None,
    )


def watch_for_model_server_service(
    namespace: str, project_name: str, project_version: str, event_handler: Callable
):
    selectors = {
        "applications.gordo.equinor.com/project-name": project_name,
        "applications.gordo.equinor.com/project-version": project_version,
        "app.kubernetes.io/component": "service",
        "app.kubernetes.io/managed-by": "gordo",
        "app.kubernetes.io/name": "model-server",
        "app.kubernetes.io/part-of": "gordo",
    }

    return watch_namespaced_services(
        event_handler=event_handler, namespace=namespace, selectors=selectors
    )
