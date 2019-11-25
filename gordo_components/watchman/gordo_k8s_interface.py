# -*- coding: utf-8 -*-
import _thread
import threading
import logging
import time
from typing import Callable, Dict, Optional

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
    except (FileNotFoundError, TypeError):
        config.load_incluster_config()  # Within cluster auth for a pod


class ThreadedListWatcher(threading.Thread):
    """Thread list-then-watch for kubernetes resources. Will restart on
    `kubernetes.client.rest.ApiException`, other exceptions causes an interrupt of the
    main thread.
    """

    def __init__(self, watched_function: Callable, event_handler: Callable, **kwargs):
        """
        A  thread list-then-watch for kubernetes resources using `watched_function`,
        invoking `event_handler` on each event emitted by the watched function.

        `watched_function` is called once in the beginning, and all returned elements
        are sent as events with `type`: `None` and `object` being the resource.

        `watched_function` is expected to be of a type which can be used by
        :py:func:`kubernetes.watch.watch.Watch.stream`, most prominently
        functions in :py:class:`kubernetes.client.apis.core_v1_api.CoreV1Api`.

        The generated thread is returned, and it is the callers responsibility to call
        `start` on it. The thread is started in daemon-mode: the entire Python program
        exits when no alive non-daemon threads are left.

        If a unhandled exception occurs in this thread it will call
        :py:func:`_thread.interrupt_main`.



        Parameters
        ----------
        watched_function: Callable
            Function use to watch for kubernetes changes
        event_handler:
            Function which will be called to handle any events emitted by
            `watched_function`
        kwargs
            Extra arguments which will be passed in to `watched_function`

        """
        self.process_event = event_handler
        self.func = watched_function
        self.kwargs = kwargs
        threading.Thread.__init__(self, daemon=True)
        # If this is set then the threads dies after the next received element. Useful
        # for testing.
        self._die_after_next = False

    def run(self):
        try:
            while True:
                try:

                    logger.debug(
                        f"Calling func {self.func} for initial list of elements"
                    )
                    initial_list_resp = self.func(**self.kwargs)
                    initial_resource_version = initial_list_resp.get(
                        "metadata", {}
                    ).get("resourceVersion", None)

                    initial_list = initial_list_resp.get("items", [])

                    logger.debug(
                        f"Initial list of objects had resource_version {initial_resource_version}"
                    )
                    logger.debug(
                        f"Initial list of objects had {len(initial_list)} items"
                    )

                    for item in initial_list:
                        self.process_event({"type": None, "object": item})

                    logger.debug("Done processing initial list of elements")

                    w = watch.Watch()
                    if initial_resource_version:
                        self.kwargs.update(
                            {"resource_version": initial_resource_version}
                        )
                    for event in w.stream(self.func, **self.kwargs):
                        if event.get("type", "").upper() == "ERROR":
                            logger.debug(
                                "Error-event received from k8s, restarting watch"
                            )
                            break  # List the objects again
                        else:
                            self.process_event(event)
                        if self._die_after_next:
                            return
                except ApiException:
                    logger.exception(
                        "Exception encountered while watching for event stream"
                    )
                    time.sleep(0.5)
                    pass
        except Exception:
            logger.exception("Got unhandled exception in k8s watching thread, exiting")
            _thread.interrupt_main()

    def die_after_next_elem(self, val=True):
        self._die_after_next = val


def watch_namespaced_custom_object(
    event_handler: Callable,
    namespace: str,
    client: Optional[kubernetes.client.apis.core_v1_api.CoreV1Api] = None,
    selectors: Optional[Dict[str, str]] = None,
) -> ThreadedListWatcher:
    """Watches changes to k8s services in a given namespace, and executed
    `event_handler` for each event, plus for all initial objects in the cluster.

    Returns the watching thread, which must be started by the caller.

    Parameters
    ----------
    event_handler : Callable
        Function which will be called on each service-event from k8s. Should take a
        single argument, which will be the k8s event for a service change. See
        `kubernetes.watch.watch.Watch.stream` for description of event structure.
    namespace: str
        Namespace to look for changes in
    client: Optional[kubernetes.client.apis.core_v1_api.CoreV1Api]
        K8s client to use to watch for changes. If None then we try to create one.
    selectors: Optional[Dict[str, str]]


    Returns
    -------
    gordo_components.watchman.gordo_k8s_interface.ThreadedListWatcher
        The watching thread, must be started by the caller.

    """

    if client is None:
        load_config()
        client = kubeclient.CustomObjectsApi()
    else:
        client = client

    kwargs = dict(
        event_handler=event_handler,
        watched_function=client.list_namespaced_custom_object,
        namespace=namespace,
        group="equinor.com",
        version="v1",
        plural="models",
    )
    if selectors:
        return ThreadedListWatcher(
            label_selector=",".join(f"{k}={v}" for k, v in selectors.items()), **kwargs
        )
    else:
        return ThreadedListWatcher(
            field_selector=f"metadata.namespace=={namespace}", **kwargs
        )
