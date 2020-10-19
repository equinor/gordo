import pytest
from mock import patch, MagicMock

from azure.core.exceptions import ResourceNotFoundError
from datetime import datetime

from azure.storage.filedatalake import PathProperties
from gordo.machine.dataset.file_system.adl2 import ADLGen2FileSystem
from gordo.machine.dataset.file_system.base import FileType, FileInfo
from azure.identity import ClientSecretCredential, InteractiveBrowserCredential


@pytest.fixture
def downloader_mock():
    return MagicMock()


@pytest.fixture
def file_client_mock(downloader_mock):
    file_client_mock = MagicMock()
    file_client_mock.download_file.return_value = downloader_mock
    return file_client_mock


@pytest.fixture
def fs_client_mock(file_client_mock):
    fs_client_mock = MagicMock()
    fs_client_mock.get_file_client.return_value = file_client_mock
    return fs_client_mock


def test_create_from_env_interactive_browser_credential():
    fs = ADLGen2FileSystem.create_from_env("dlaccount", "fs", interactive=True)
    assert isinstance(fs.file_system_client.credential, InteractiveBrowserCredential)
    assert fs.account_name == "dlaccount"
    assert fs.file_system_name == "fs"


def test_create_from_env_client_secret_credential():
    with patch("os.environ.get") as get_mock:
        get_mock.return_value = "tenant_id:client_id:client_secret"
        fs = ADLGen2FileSystem.create_from_env("dlaccount", "fs", interactive=False)
        assert isinstance(fs.file_system_client.credential, ClientSecretCredential)
        assert fs.account_name == "dlaccount"
        assert fs.file_system_name == "fs"


def test_open(downloader_mock, fs_client_mock):
    downloader_mock.readall.return_value = b"\x7fELF\x02"
    fs = ADLGen2FileSystem(fs_client_mock, "dlaccount", "fs")
    with fs.open("/path/to/file", "rb") as f:
        assert f.read() == b"\x7fELF\x02"
    fs_client_mock.get_file_client.assert_called_once_with("/path/to/file")


def test_not_exists(fs_client_mock, file_client_mock):
    file_client_mock.get_file_properties.side_effect = ResourceNotFoundError
    fs = ADLGen2FileSystem(fs_client_mock, "dlaccount", "fs")
    assert not fs.exists("/path/to/file")


def test_exists_file(fs_client_mock, file_client_mock):
    last_modified = datetime(2020, 9, 17, 0, 0, 0, 0)
    creation_time = datetime(2019, 4, 10, 0, 0, 0, 0)
    file_client_mock.get_file_properties.return_value = {
        "size": 1000,
        "content_settings": {"content_type": "application/json"},
        "last_modified": last_modified,
        "creation_time": creation_time,
    }
    fs = ADLGen2FileSystem(fs_client_mock, "dlaccount", "fs")
    info = fs.info("/path/to/file.json")
    assert info.size == 1000
    assert info.file_type == FileType.FILE
    assert info.access_time is None
    assert info.modify_time == last_modified
    assert info.create_time == creation_time
    assert fs.exists("/path/to/file.json")
    assert fs.isfile("/path/to/file.json")
    assert not fs.isdir("/path/to/file.json")
    assert fs_client_mock.get_file_client.call_count == 4


def test_exists_directory(fs_client_mock, file_client_mock):
    last_modified = datetime(2020, 9, 16, 0, 0, 0, 0)
    creation_time = datetime(2019, 4, 10, 0, 0, 0, 0)
    file_client_mock.get_file_properties.return_value = {
        "size": 0,
        "content_settings": {"content_type": None},
        "last_modified": last_modified,
        "creation_time": creation_time,
    }
    fs = ADLGen2FileSystem(fs_client_mock, "dlaccount", "fs")
    info = fs.info("/path/to")
    assert info.size == 0
    assert info.file_type == FileType.DIRECTORY
    assert info.access_time is None
    assert info.modify_time == last_modified
    assert info.create_time == creation_time
    assert fs.exists("/path/to")
    assert not fs.isfile("/path/to")
    assert fs.isdir("/path/to")
    assert fs_client_mock.get_file_client.call_count == 4


def create_path_properties(name: str, is_directory: bool, content_length: int = 0):
    properties = PathProperties()
    properties.name = name
    properties.is_directory = is_directory
    properties.content_length = content_length
    return properties


def test_ls_without_info(fs_client_mock):
    fs_client_mock.get_paths.return_value = [
        create_path_properties(name="/path/to", is_directory=True),
        create_path_properties(
            name="/path/file.json", is_directory=False, content_length=12430
        ),
    ]
    fs = ADLGen2FileSystem(fs_client_mock, "dlaccount", "fs")
    result = list(fs.ls("/path", with_info=False))
    assert result == [("/path/to", None), ("/path/file.json", None)]
    fs_client_mock.get_paths.assert_called_once_with("/path", recursive=False)


def test_ls_with_info(fs_client_mock):
    fs_client_mock.get_paths.return_value = [
        create_path_properties(name="/path/to", is_directory=True),
        create_path_properties(
            name="/path/file.json", is_directory=False, content_length=12430
        ),
    ]
    fs = ADLGen2FileSystem(fs_client_mock, "dlaccount", "fs")
    result = list(fs.ls("/path", with_info=True))
    assert result == [
        (
            "/path/to",
            FileInfo(
                file_type=FileType.DIRECTORY,
                size=0,
                access_time=None,
                modify_time=None,
                create_time=None,
            ),
        ),
        (
            "/path/file.json",
            FileInfo(
                file_type=FileType.FILE,
                size=12430,
                access_time=None,
                modify_time=None,
                create_time=None,
            ),
        ),
    ]
    fs_client_mock.get_paths.assert_called_once_with("/path", recursive=False)


def test_walk_without_info(fs_client_mock):
    fs_client_mock.get_paths.return_value = [
        create_path_properties(name="/path/to", is_directory=True),
        create_path_properties(
            name="/path/to/file.json", is_directory=False, content_length=12430
        ),
    ]
    fs = ADLGen2FileSystem(fs_client_mock, "dlaccount", "fs")
    result = list(fs.walk("/path", with_info=False))
    assert result == [("/path/to", None), ("/path/to/file.json", None)]
    fs_client_mock.get_paths.assert_called_once_with("/path")


def test_walk_with_info(fs_client_mock):
    fs_client_mock.get_paths.return_value = [
        create_path_properties(name="/path/to", is_directory=True),
        create_path_properties(
            name="/path/to/file.json", is_directory=False, content_length=12430
        ),
    ]
    fs = ADLGen2FileSystem(fs_client_mock, "dlaccount", "fs")
    result = list(fs.walk("/path", with_info=True))
    assert result == [
        (
            "/path/to",
            FileInfo(
                file_type=FileType.DIRECTORY,
                size=0,
                access_time=None,
                modify_time=None,
                create_time=None,
            ),
        ),
        (
            "/path/to/file.json",
            FileInfo(
                file_type=FileType.FILE,
                size=12430,
                access_time=None,
                modify_time=None,
                create_time=None,
            ),
        ),
    ]
    fs_client_mock.get_paths.assert_called_once_with("/path")
