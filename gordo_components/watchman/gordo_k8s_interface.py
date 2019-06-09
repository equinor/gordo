# -*- coding: utf-8 -*-
import _thread
import threading
import logging
from typing import Callable, Dict

import kubernetes
from kubernetes import client as kubeclient, config, watch
from kubernetes.client.rest import ApiException


logger = logging.getLogger(__name__)


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


class ThreadedWatcher(threading.Thread):
    """Thread watching for changes in kubernetes. Will restart on
    `kubernetes.client.rest.ApiException`, other exceptions causes an interrupt of the
    main thread.
    """

    def __init__(self, watched_function, processer: Callable, **kwargs):
        self.process_event = processer
        self.func = watched_function
        self.kwargs = kwargs
        threading.Thread.__init__(self, daemon=True)

    def run(self):
        try:
            while True:
                try:
                    w = watch.Watch()
                    for event in w.stream(self.func, **self.kwargs):
                        self.process_event(event)
                except ApiException:
                    logger.error(
                        "Exception encountered while watching for event stream"
                    )
                    pass
        except Exception:
            logger.exception("Got unhandled exception in k8s watching thread, exiting")
            _thread.interrupt_main()


def watch_service(
    processor: Callable,
    namespace: str,
    client: kubernetes.client.apis.core_v1_api.CoreV1Api = None,
    selectors: Dict[str, str] = None,
):
    """Watches services in a namespace, and executed processor for each event. Returns
    the watching thread."""

    if client is None:
        load_config()
        client = kubeclient.CoreV1Api()
    else:
        client = client
    if selectors:
        return ThreadedWatcher(
            watched_function=client.list_service_for_all_namespaces,
            processer=processor,
            field_selector=f"metadata.namespace=={namespace}",
            label_selector=",".join(f"{k}={v}" for k, v in selectors.items()),
        )
    else:
        return ThreadedWatcher(
            client.read_namespaced_service, processor, namespace=namespace
        )
