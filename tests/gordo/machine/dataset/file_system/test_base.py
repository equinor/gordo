from gordo.machine.dataset.file_system.base import default_join


def test_default_join():
    assert default_join("/path/to/file", "") == "/path/to/file"
    assert default_join("") == ""
    assert default_join() == ""
