import logging

import time

import pytest
from mock import MagicMock

from gordo_components.watchman.gordo_k8s_interface import watch_namespaced_custom_object

logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "selectors",
    (
        None,
        {
            "app.kubernetes.io/component": "service",
            "app.kubernetes.io/managed-by": "gordo",
        },
    ),
)
@pytest.mark.parametrize(
    "error410", (True, False),
)
def test_watch_service_basic(mocker, selectors, error410):

    mock_client = MagicMock()
    # This sets it up list_namespaced_custom_object to return  different lists of
    # elements the two first times it is called
    list_events = [
        {
            "items": [
                {"type": "added", "value": "firstValueFirstListing"},
                {"type": "added", "value": "secondValueFirstListing"},
            ]
        },
        {
            "items": [
                {"type": "added", "value": "firstValueSecondListing"},
                {"type": "added", "value": "secondValueSecondListing"},
            ]
        },
    ]

    # This is the list of elements which will be returned when it starts watching
    stream_events = [
        [
            {"object": {"type": "added", "value": "firstValueStreamed"}},
            {"object": {"type": "added", "value": "secondValueStreamed"}},
        ]
    ]

    if error410:
        stream_events = [[{"type": "ERROR", "object": {}}]] + stream_events

    mock_client.list_namespaced_custom_object = MagicMock(side_effect=list_events)

    receiver = MagicMock()
    mocker.patch(
        "kubernetes.watch.watch.Watch.stream", MagicMock(side_effect=stream_events)
    )

    if selectors:
        watch_thread = watch_namespaced_custom_object(
            event_handler=receiver,
            namespace="somenamespace",
            client=mock_client,
            selectors=selectors,
        )
    else:
        watch_thread = watch_namespaced_custom_object(
            event_handler=receiver, namespace="somenamespace", client=mock_client,
        )

    # Ensure watch_thread dies after fetching its first element from the
    # watch-stream.
    watch_thread.die_after_next_elem()
    watch_thread.start()
    # Give the background thread some time to run, up to 3 seconds.
    max_ms_to_wait = 3000
    while receiver.call_count < 3 and max_ms_to_wait > 0:
        time.sleep(0.01)
        max_ms_to_wait = max_ms_to_wait - 1

    assert receiver.call_args_list[0].args[0] == {
        "object": {"type": "added", "value": "firstValueFirstListing"},
        "type": None,
    }
    assert receiver.call_args_list[1].args[0] == {
        "object": {"type": "added", "value": "secondValueFirstListing"},
        "type": None,
    }
    if error410:
        assert receiver.call_args_list[2].args[0] == {
            "object": {"type": "added", "value": "firstValueSecondListing"},
            "type": None,
        }
        assert receiver.call_args_list[3].args[0] == {
            "object": {"type": "added", "value": "secondValueSecondListing"},
            "type": None,
        }
        assert receiver.call_args_list[4].args[0] == {
            "object": {"type": "added", "value": "firstValueStreamed"}
        }

    else:
        assert receiver.call_args_list[2].args[0] == {
            "object": {"type": "added", "value": "firstValueStreamed"},
        }
    # Ensure watch_thread is done
    watch_thread.join()
