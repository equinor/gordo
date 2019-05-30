import logging

import time
from mock import MagicMock

from gordo_components.watchman.gordo_k8s_interface import watch_namespaced_services

logger = logging.getLogger(__name__)


def test_watch_service(mocker):
    mock_client = MagicMock()

    last_seen_value = None

    def assign_to_last_sendt_var(x):
        nonlocal last_seen_value
        last_seen_value = x

    called_count = 0

    def return_increasing_nr_and_sleep(**_kwargs):
        nonlocal called_count
        time.sleep(called_count)
        called_count = called_count + 1
        return called_count

    def watch_stream_mock(_self, func, **_kwargs):
        return [func()]

    mocker.patch("kubernetes.watch.watch.Watch.stream", watch_stream_mock)
    mock_client.list_service_for_all_namespaces = return_increasing_nr_and_sleep

    watch_thread = watch_namespaced_services(
        event_handler=assign_to_last_sendt_var,
        namespace="somenamespace",
        client=mock_client,
    )
    # Ensure watch_thread dies after its first element so we dont have dangling threads
    # laying around
    watch_thread.die_after_next_elem()
    watch_thread.start()
    # Give the background thread some time to run, up to 3 seconds.
    max_ms_to_wait = 3000
    while last_seen_value is None and max_ms_to_wait > 0:
        time.sleep(0.01)
        max_ms_to_wait = max_ms_to_wait - 1
    assert last_seen_value is not None
    # Ensure watch_thread is done
    watch_thread.join()

    # And again, but this time with selectors
    watch_thread = watch_namespaced_services(
        event_handler=assign_to_last_sendt_var,
        namespace="somenamespace",
        client=mock_client,
        selectors={
            "app.kubernetes.io/component": "service",
            "app.kubernetes.io/managed-by": "gordo",
        },
    )
    # Ensure watch_thread dies after its first element so we dont have dangling threads
    # laying around
    watch_thread.die_after_next_elem()
    watch_thread.start()
    # Give the background thread some time to run, up to 3 seconds.
    max_ms_to_wait = 3000
    while last_seen_value is None and max_ms_to_wait > 0:
        time.sleep(0.01)
        max_ms_to_wait = max_ms_to_wait - 1
    assert last_seen_value is not None
    # Ensure watch_thread is done
    watch_thread.join()
