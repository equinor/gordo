import io
from gordo_components.client.utils import EndpointMetadata


def test_endpoint_metadata():

    data = dict(
        key1="value1",
        key2=[dict(subkey1="subvalue1")],
        key3=dict(subkey2="subkey2"),
        key4=[1, 2],
        key5={"hyphen-key": 2},
        key6=dict(underscore_key=1),
    )

    epm = EndpointMetadata(data)
    assert epm.key1 == "value1"
    assert epm.key2[0].subkey1 == "subvalue1"
    assert epm.key3.subkey2 == "subkey2"
    assert epm.key4[0] == 1
    assert len(epm.key4) == 2
    assert epm.key4[0] == 1
    assert epm.key4[1] == 2
    assert epm.key5.hyphen_key == 2
    assert epm.key6.underscore_key == 1

    # Access item raw by calling it.
    assert epm.key5() == {"hyphen-key": 2}
