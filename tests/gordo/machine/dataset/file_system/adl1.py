import pytest
from mock import patch, Mock
from io import BytesIO, TextIOWrapper

from gordo.machine.dataset.file_system.adl1 import ADLGen1FileSystem
from gordo.machine.dataset.file_system.base import FileType


@pytest.fixture
def auth_mock():
    with patch("azure.datalake.store.lib.auth") as auth:
        auth.return_value = "123"
        yield auth


@pytest.fixture
def adl_client_mock():
    with patch("azure.datalake.store.core.AzureDLFileSystem") as adl_client:
        adl_client.open = Mock(return_value=BytesIO())
        adl_client.info = Mock()
        adl_client.exists = Mock()
        yield adl_client


def test_create_from_env_interactive(auth_mock, adl_client_mock):
    ADLGen1FileSystem.create_from_env("dlstore", interactive=True)
    auth_mock.assert_called_once_with()
    adl_client_mock.assert_called_once_with("123", store_name="dlstore")


def test_create_from_env_with_dl_service_auth(auth_mock, adl_client_mock):
    ADLGen1FileSystem.create_from_env("dlstore", dl_service_auth="tenant_id:client_id:client_secret")
    auth_mock.assert_called_once_with(
        tenant_id="tenant_id",
        client_id="client_id",
        client_secret="client_secret",
        resource="https://datalake.azure.net/",
    )
    adl_client_mock.assert_called_once_with("123", store_name="dlstore")


def test_create_from_env_with_dl_service_auth_env(auth_mock, adl_client_mock):
    with patch("os.environ.get") as get:
        get.return_value = "tenant_id:client_id:client_secret"
        ADLGen1FileSystem.create_from_env("dlstore")
        auth_mock.assert_called_once_with(
            tenant_id="tenant_id",
            client_id="client_id",
            client_secret="client_secret",
            resource="https://datalake.azure.net/",
        )
        adl_client_mock.assert_called_once_with("123", store_name="dlstore")


def test_create_from_env_with_invalid_dl_service(auth_mock, adl_client_mock):
    with pytest.raises(ValueError):
        ADLGen1FileSystem.create_from_env("dlstore", dl_service_auth="tenant_id:client_id")


def test_create_from_env_with_empty_dl_service_auth_env(auth_mock, adl_client_mock):
    with patch("os.environ.get") as get:
        get.return_value = None
        with pytest.raises(ValueError):
            ADLGen1FileSystem.create_from_env("dlstore")


def test_open_in_bin_mode(adl_client_mock):
    fs = ADLGen1FileSystem(adl_client_mock)
    f = fs.open("/path/to/file.json", mode="rb")
    adl_client_mock.open.assert_called_once_with("/path/to/file.json", mode="rb")
    assert isinstance(f, BytesIO)


def test_open_in_text_mode(adl_client_mock):
    fs = ADLGen1FileSystem(adl_client_mock)
    f = fs.open("/path/to/file.json", mode="r")
    adl_client_mock.open.assert_called_once_with("/path/to/file.json", mode="rb")
    assert isinstance(f, TextIOWrapper)


def test_exists(adl_client_mock):
    adl_client_mock.exists.return_value = True
    fs = ADLGen1FileSystem(adl_client_mock)
    assert fs.exists("/path/to/file.json")
    adl_client_mock.exists.assert_called_once_with("/path/to/file.json")


def test_isfile(adl_client_mock):
    adl_client_mock.info.return_value = {"type": "FILE"}
    fs = ADLGen1FileSystem(adl_client_mock)
    assert fs.isfile("/path/to/file.json")
    adl_client_mock.info.assert_called_once_with("/path/to/file.json")


def test_isdir(adl_client_mock):
    adl_client_mock.info.return_value = {"type": "DIRECTORY"}
    fs = ADLGen1FileSystem(adl_client_mock)
    assert fs.isdir("/path/to/file.json")
    adl_client_mock.info.assert_called_once_with("/path/to/file.json")


def test_info_file(adl_client_mock):
    adl_client_mock.info.return_value = {
        "type": "FILE",
        "length": 304254,
        "accessTime": 1599631062424,
        "modificationTime": 1599631097160,
    }
    fs = ADLGen1FileSystem(adl_client_mock)
    info = fs.info("/path/to/file.json")
    adl_client_mock.info.assert_called_once_with("/path/to/file.json")
    assert info.file_type == FileType.FILE
    assert info.size == 304254
    assert info.access_time.isoformat() == "2020-09-09T05:57:42.424000"
    assert info.modify_time.isoformat() == "2020-09-09T05:58:17.160000"


def test_info_directory(adl_client_mock):
    adl_client_mock.info.return_value = {
        "type": "DIRECTORY",
        "length": 0,
    }
    fs = ADLGen1FileSystem(adl_client_mock)
    info = fs.info("/path/to/file.json")
    adl_client_mock.info.assert_called_once_with("/path/to/file.json")
    assert info.file_type == FileType.DIRECTORY
    assert info.size == 0
    assert info.access_time is None
    assert info.modify_time is None
