# -*- coding: utf-8 -*-

from contextlib import contextmanager, ExitStack
from unittest import mock

import yaml
from kubernetes.client import models

"""
Various mocking functions for interacting with the kubernetes API
"""


def mock_kubernetes_read_namespaced_service(*_args, **_kwargs):
    """
    Represents the mocked output of kubernetes.client.read_namespaced_service
    """
    return models.V1Service(
        spec=models.V1ServiceSpec(selector={"app": "some-project-name"})
    )


def mock_kubernetes_read_namespaced_pod_log(*_args, **_kwargs):
    """
    Represents the mocked output of kubernetes.client.read_namespaced_pod_log
    """
    return "some example log\noutput data goes\nhere."


def mock_kubernetes_list_namespaced_pod(*args, **kwargs):
    """
    Represents the mocked ouptut of kubernetes.client.list_namespaced_pod
    """
    return models.V1PodList(
        items=[mock_kubernetes_read_namespaced_pod(*args, **kwargs)]
    )


def mock_kubernetes_list_namespaced_custom_object(*_args, **_kwargs):
    """
    Represents the mocked output of kubernetes.client.list_namespaced_custom_object
    specifically for an Argo Workflow custom object
    """
    workflows = """
    items:
      - metadata:
          creationTimestamp: '2019-03-14T15:27:27.297708'
          generateName: gordo-test-1234.4-
          labels:
            applications.gordo.equinor.com/project-name: gordo-test
            applications.gordo.equinor.com/project-version: 1
        status:
          nodes:
            gordo-test-pod-name-1234:
              displayName: build-model
              inputs:
                parameters:
                  - name: machine-name
                    value: test-machine-name
    """
    return yaml.safe_load(workflows)


def mock_kubernetes_read_namespaced_pod(*_args, **_kwargs):
    """
    Represents the mocked output of kubernetes.client.read_namespaced_pod
    """
    return models.V1Pod(
        metadata=models.V1ObjectMeta(
            namespace="default",
            name="gordo-test-pod-name-1234",
            labels={"app": "gordo-model-builder"},
        ),
        status=models.V1PodStatus(phase="Running"),
        spec=models.V1PodSpec(
            containers=[
                models.V1Container(
                    name="some-generated-test-container-name",
                    env=[models.V1EnvVar(name="MODEL_NAME", value="test-machine-name")],
                )
            ]
        ),
    )


@contextmanager
def mocked_kubernetes():
    """
    Mock the needed kubernetes interactions for tests requiring kubernetes API
    interactions.
    """
    contexts = [
        mock.patch(
            "kubernetes.client.CoreV1Api.read_namespaced_pod",
            side_effect=mock_kubernetes_read_namespaced_pod,
        ),
        mock.patch(
            "kubernetes.client.CustomObjectsApi.list_namespaced_custom_object",
            side_effect=mock_kubernetes_list_namespaced_custom_object,
        ),
        mock.patch(
            "kubernetes.config.load_kube_config",
            side_effect=lambda *args, **kwargs: None,
        ),
        mock.patch(
            "kubernetes.client.CoreV1Api.read_namespaced_pod_log",
            side_effect=mock_kubernetes_read_namespaced_pod_log,
        ),
        mock.patch(
            "kubernetes.client.CoreV1Api.read_namespaced_service",
            side_effect=mock_kubernetes_read_namespaced_service,
        ),
        mock.patch(
            "kubernetes.client.CoreV1Api.list_namespaced_pod",
            side_effect=mock_kubernetes_list_namespaced_pod,
        ),
    ]

    with ExitStack() as stack:
        for ctx in contexts:
            stack.enter_context(ctx)
        yield
