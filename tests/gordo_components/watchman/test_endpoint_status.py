import logging

from apscheduler.schedulers.blocking import BlockingScheduler
from mock import MagicMock

from gordo_components.watchman.endpoints_status import EndpointStatuses

logger = logging.getLogger(__name__)


def test_EndpointStatuses(mocker):
    """Tests the main workflow of endpointStatuses.

    We mock away at two places,
    gordo_components.watchman.endpoints_status.watch_for_model_server_service
    which listens for kubernetes events is mocked away, and we construct the events
    manually.

    And the job-scheduler is never actually started, so we dont proceed to do any of
    the jobs, we just check that they are added/removed as desired.

    """

    # We will never call start on the scheduler, so we wont actually do any of the
    # scheduled jobs|
    scheduler = BlockingScheduler()
    project_name = "super_project"
    project_version = "101"
    namespace = "somenamespace"
    host = "localhost"
    target_names = ["target 1", "target 2"]
    mocked_watch = mocker.patch(
        "gordo_components.watchman.endpoints_status.watch_for_model_server_service"
    )
    eps = EndpointStatuses(
        scheduler=scheduler,
        project_name=project_name,
        ambassador_host=host,
        model_names=target_names,
        project_version=project_version,
        namespace=namespace,
    )

    assert namespace == mocked_watch.call_args[1]["namespace"]
    assert project_name == mocked_watch.call_args[1]["project_name"]
    assert project_version == mocked_watch.call_args[1]["project_version"]
    event_handler = mocked_watch.call_args[1]["event_handler"]

    # Before receiving any events we only have the targets in `target_names`
    cur_status = eps.statuses()
    assert set([ep["target"] for ep in cur_status]) == set(target_names)

    # And none of them are healthy
    assert all([ep["healthy"] is False for ep in cur_status])

    # Lets start adding some events.

    # Target 1 is online!
    # We make a fake event
    mock_event_obj = MagicMock()
    mock_event_obj.metadata = MagicMock()
    mock_event_obj.metadata.labels = {
        "applications.gordo.equinor.com/model-name": "target 1"
    }
    # And we let the caller know about it
    event_handler({"type": "ADDED", "object": mock_event_obj})

    # The job to update target 1 is added to the joblist
    jobs = scheduler.get_jobs()
    assert len(jobs) == 1
    assert scheduler.get_job("update_model_metadata_target 1") is not None

    # Target 2 is up as well
    mock_event_obj.metadata.labels[
        "applications.gordo.equinor.com/model-name"
    ] = "target 2"
    event_handler({"type": "ADDED", "object": mock_event_obj})

    jobs = scheduler.get_jobs()
    assert len(jobs) == 2
    assert scheduler.get_job("update_model_metadata_target 2") is not None

    # Oida, target 1 seems to be removed!
    mock_event_obj.metadata.labels = {
        "applications.gordo.equinor.com/model-name": "target 1"
    }

    event_handler({"type": "DELETED", "object": mock_event_obj})

    jobs = scheduler.get_jobs()
    assert len(jobs) == 1
    assert scheduler.get_job("update_model_metadata_target 1") is None
    assert scheduler.get_job("update_model_metadata_target 2") is not None
