from gordo_components.client.utils import EndpointMetadata
from gordo_components.dataset.sensor_tag import SensorTag


def test_endpoint_metadata():
    data = {
        "endpoint-metadata": {
            "metadata": {
                "name": "test-project-target-a",
                "dataset": {
                    "resolution": "10T",
                    "tag_list": [],
                    "target_tag_list": [{"asset": "foo", "name": "bar"}],
                },
                "model": {"model-offset": 1},
            }
        },
        "healthy": True,
        "endpoint": "/gordo/v0/test-project/test-project-target-a",
    }

    epm = EndpointMetadata(data)
    assert epm.name == "test-project-target-a"
    assert epm.endpoint == "/gordo/v0/test-project/test-project-target-a"
    assert epm.tag_list == []
    assert epm.target_tag_list == [SensorTag(name="bar", asset="foo")]
    assert epm.resolution == "10T"
    assert epm.model_offset == 1
    assert epm.healthy is True
    assert epm.raw_metadata() == data
