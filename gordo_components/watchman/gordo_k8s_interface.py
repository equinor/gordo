# -*- coding: utf-8 -*-

from typing import Optional, Dict, List

from kubernetes import client as kubeclient, config
from kubernetes.client.rest import ApiException
from kubernetes.client.models.v1_pod_status import V1PodStatus
from kubernetes.client.models.v1_pod import V1Pod


def load_config():
    """
    Attempts to load cluster config from $HOME/.kube/config with a fallback
    to the incluster
    Returns
    -------

    """
    try:
        config.load_kube_config()  # Local .kube/config
    except FileNotFoundError:
        config.load_incluster_config()  # Within cluster auth for a pod


def list_model_builders(
    namespace: str,
    project_name: str,
    project_version: str,
    client: kubeclient.CustomObjectsApi = None,
    selectors: Optional[Dict[str, str]] = None,
):
    """
        Get a list of pods which were responsible for building models for a given
        project and version

        Parameters
        ----------
        namespace: str
            Namespace to operate in
        project_name: str
            Project name
        project_version: str
            Project version
        client: kubernetes.client.CustomObjectApi
            The client to use in selecting custom objects from Kubernetes
        selectors: Optional[Dict[str, str]]
            A mapping of key value pairs representing the label in the workflow
            to match to a value, if not set then match on project_name and project_version
        """

    # Set default selectors if not provided.
    if selectors is None:
        selectors = {
            "applications.gordo.equinor.com/project-name": project_name,
            "applications.gordo.equinor.com/project-version": project_version,
        }
    selectors.update({"app": "gordo-model-builder"})

    if client is None:
        load_config()
        client = kubeclient.CoreV1Api()

    return [
        ModelBuilderPod(pod, client=client)
        for pod in client.list_namespaced_pod(
            namespace=namespace,
            label_selector=",".join(f"{k}={v}" for k, v in selectors.items()),
        ).items
    ]


class Service:

    pods: List["Pod"]
    name: str
    namespace: str
    client: kubeclient.CoreV1Api

    def __init__(self, namespace: str, name: str, client: kubeclient.CoreV1Api = None):
        """
        Construct a Gordo interface to a k8s service, and the pods it services.
        Creates the list of pods on initialization, and does not refresh it

        Parameters
        ----------
        namespace: str
            Name of the namespace to look in for the service
        name: str
            Name of the service
        client: Optional[kubernetes.client.CoreV1Api]
            Client to use, otherwise will try to load environment config to create one.
        """
        self.namespace = namespace
        self.name = name

        if client is None:
            load_config()
            self.client = kubeclient.CoreV1Api()
        else:
            self.client = client

        # Reference to the kubernetes service itself.
        self._service = self.client.read_namespaced_service(
            name=self.name, namespace=self.namespace
        )

        # Create a pod for each one in this service
        self.pods = [
            Pod(pod, client=self.client)
            for pod in self.client.list_namespaced_pod(
                namespace=self.namespace,
                label_selector=",".join(
                    f"{k}={v}" for k, v in self._service.spec.selector.items()
                ),
            ).items
        ]

    def __len__(self):
        """Represents the number of pods in the service"""
        return len(self.pods)

    @property
    def status(self):
        """
        Percentage of healthy pods in this service

        Returns
        -------
        float:
            percentage of healthy pods in this service
        """
        if len(self) > 0:
            return sum(1 for p in self.pods if p.is_healthy) / len(self)
        else:
            return 0

    def logs(self, limit=40) -> Dict[str, str]:
        """
        Get the logs from all pods related to this service

        Parameters
        ----------
        limit: int
            How many lines to retrieve from each pod

        Returns
        -------
        Dict[str, str]
            Mapping where each key is the pod name, and value of the log output
        """
        return {p.name: p.logs(limit) for p in self.pods}


class Pod:

    pod: V1Pod

    def __init__(self, pod: V1Pod, client: Optional[kubeclient.CoreV1Api] = None):
        """
        Simple helper class around kubernetes.client.models.v1_pod.V1Pod for essential
        gordo operations by Watchman

        Parameters
        ----------
        pod: kubernetes.models.V1Pod
        client: kubernetes.client.CoreV1Api
        """
        if client is None:
            load_config()
            self.client = kubeclient.CoreV1Api()
        else:
            self.client = client

        self.pod = pod

    def logs(self, limit: int = 40) -> str:
        """
        Return the tail of logs from this pod

        Parameters
        ----------
        limit: int
            Number of lines from the end of the logs

        Returns
        -------
        logs: str
        """
        # Required arguments, potentially requiring container specification (main)
        kwargs = dict(
            name=self.pod.metadata.name,
            namespace=self.pod.metadata.namespace,
            tail_lines=limit,
        )

        try:
            log = self.client.read_namespaced_pod_log(**kwargs)
        except ApiException:
            kwargs["container"] = "main"
            log = self.client.read_namespaced_pod_log(**kwargs)
        return log

    def __repr__(self):
        return (
            f"Pod<namespace={self.pod.metadata.namespace}, "
            f"name={self.pod.metadata.name}, healthy={self.is_healthy}>"
        )

    @property
    def name(self):
        """Name of the pod"""
        return self.pod.metadata.name

    @property
    def is_healthy(self) -> bool:
        """
        Determine if the current pod is in a healthy state or has completed
        successfully.

        Returns
        -------
        bool:
            Indication of success
        """
        return self.pod.status.phase in ["Succeeded", "Completed", "Running"]

    @property
    def status(self) -> V1PodStatus:
        """
        Return the pods' V1PodStatus object
        """
        return self.pod.status


class ModelBuilderPod(Pod):
    """
    A pod specific to building a model for a given target/machine.
    """

    def __init__(self, pod: V1Pod, *args, **kwargs):
        """
        Get a Gordo ModelBuilderPod representation from the raw kubernetes V1Pod object

        This specific one is expected to have spec.containers[0].env which contains
        a 'MODEL_NAME' kubernetes.models.V1EnvVar

        Parameters
        ----------
        pod: kubernetes.client.models.v1_pod.V1Pod

        Returns
        -------
        ModelBuilderPod:
            Gordo pod interface representing a Model builder pod.
        """
        if pod.metadata.labels["app"] != "gordo-model-builder":
            raise ValueError(f"This pod does not appear to be a ModelBuilder pod.")

        for envar in pod.spec.containers[0].env:
            if envar.name == "MODEL_NAME":
                self.target_name = envar.value
                break
        else:
            raise ValueError(
                f"Unable to find the MODEL_NAME in this pod's environment spec"
            )

        super().__init__(pod, *args, **kwargs)

    def __repr__(self):
        return (
            f"ModelBuilderPod<namespace={self.pod.metadata.namespace}, name={self.pod.metadata.name}, "
            f"target_name={self.target_name} healthy={self.is_healthy}>"
        )
