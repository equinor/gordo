import pytest
from mock import patch, Mock
from io import BytesIO, TextIOWrapper
from copy import copy
from datetime import datetime

from gordo.machine.dataset.file_system.adl1 import ADLGen1FileSystem
from gordo.machine.dataset.file_system.base import FileType, FileInfo


@pytest.fixture
def auth_mock():
    with patch("azure.datalake.store.lib.auth") as auth:
        auth.return_value = "123"
        yield auth


@pytest.fixture
def dirs_tree():
    return {
        "/path": (
            {
                "length": 0,
                "type": "DIRECTORY",
                "accessTime": 1592124051659,
                "modificationTime": 1592127183911,
            },
            ["/path/to", "/path/out.json"],
        ),
        "/path/to": ({"length": 0, "type": "DIRECTORY",}, ["/path/to/file.json"]),
        "/path/to/file.json": (
            {
                "length": 142453,
                "type": "FILE",
                "accessTime": 1599603699143,
                "modificationTime": 1599604856564,
            },
            [],
        ),
        "/path/out.json": (
            {
                "length": 983432,
                "type": "FILE",
                "accessTime": 1592127206477,
                "modificationTime": 1599636974996,
            },
            [],
        ),
    }


@pytest.fixture
def adl_client_mock(dirs_tree):
    def ls_side_effect(path, **kwargs):
        _, children = dirs_tree[path]
        detail = kwargs.get("detail", False)
        if detail:
            ls_result = []
            for child in children:
                info = copy(dirs_tree[child][0])
                info["name"] = child
                ls_result.append(info)
            return ls_result
        else:
            return children

    with patch("azure.datalake.store.core.AzureDLFileSystem") as adl_client:
        adl_client.open = Mock(return_value=BytesIO())
        adl_client.info = Mock()
        adl_client.exists = Mock()
        adl_client.ls = Mock(side_effect=ls_side_effect)
        yield adl_client


def test_ls_side_effect(adl_client_mock):
    assert adl_client_mock.ls("/path") == ["/path/to", "/path/out.json"]
    assert adl_client_mock.ls("/path", detail=True) == [
        {"length": 0, "type": "DIRECTORY", "name": "/path/to"},
        {
            "length": 983432,
            "type": "FILE",
            "accessTime": 1592127206477,
            "modificationTime": 1599636974996,
            "name": "/path/out.json",
        },
    ]


def test_create_from_env_interactive(auth_mock, adl_client_mock):
    ADLGen1FileSystem.create_from_env("dlstore", interactive=True)
    auth_mock.assert_called_once_with()
    adl_client_mock.assert_called_once_with("123", store_name="dlstore")


def test_create_from_env_with_dl_service_auth(auth_mock, adl_client_mock):
    ADLGen1FileSystem.create_from_env(
        "dlstore", dl_service_auth="tenant_id:client_id:client_secret"
    )
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
        ADLGen1FileSystem.create_from_env(
            "dlstore", dl_service_auth="tenant_id:client_id"
        )


def test_create_from_env_with_empty_dl_service_auth_env(auth_mock, adl_client_mock):
    with patch("os.environ.get") as get:
        get.return_value = None
        with pytest.raises(ValueError):
            ADLGen1FileSystem.create_from_env("dlstore")


def test_open_in_bin_mode(adl_client_mock):
    fs = ADLGen1FileSystem(adl_client_mock, store_name="dlstore")
    f = fs.open("/path/to/file.json", mode="rb")
    adl_client_mock.open.assert_called_once_with("/path/to/file.json", mode="rb")
    assert isinstance(f, BytesIO)


def test_open_in_text_mode(adl_client_mock):
    fs = ADLGen1FileSystem(adl_client_mock, store_name="dlstore")
    f = fs.open("/path/to/file.json", mode="r")
    adl_client_mock.open.assert_called_once_with("/path/to/file.json", mode="rb")
    assert isinstance(f, TextIOWrapper)


def test_exists(adl_client_mock):
    adl_client_mock.exists.return_value = True
    fs = ADLGen1FileSystem(adl_client_mock, store_name="dlstore")
    assert fs.exists("/path/to/file.json")
    adl_client_mock.exists.assert_called_once_with("/path/to/file.json")


def test_isfile(adl_client_mock):
    adl_client_mock.info.return_value = {"type": "FILE"}
    fs = ADLGen1FileSystem(adl_client_mock, store_name="dlstore")
    assert fs.isfile("/path/to/file.json")
    adl_client_mock.info.assert_called_once_with("/path/to/file.json")


def test_isdir(adl_client_mock):
    adl_client_mock.info.return_value = {"type": "DIRECTORY"}
    fs = ADLGen1FileSystem(adl_client_mock, store_name="dlstore")
    assert fs.isdir("/path/to/file.json")
    adl_client_mock.info.assert_called_once_with("/path/to/file.json")


def test_info_file(adl_client_mock):
    adl_client_mock.info.return_value = {
        "type": "FILE",
        "length": 304254,
        "accessTime": 1599631062424,
        "modificationTime": 1599631097160,
    }
    fs = ADLGen1FileSystem(adl_client_mock, store_name="dlstore")
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
    fs = ADLGen1FileSystem(adl_client_mock, store_name="dlstore")
    info = fs.info("/path/to/file.json")
    adl_client_mock.info.assert_called_once_with("/path/to/file.json")
    assert info.file_type == FileType.DIRECTORY
    assert info.size == 0
    assert info.access_time is None
    assert info.modify_time is None


def test_ls_without_info(adl_client_mock):
    fs = ADLGen1FileSystem(adl_client_mock, store_name="dlstore")
    result = list(fs.ls("/path", with_info=False))
    assert result == [("/path/to", None), ("/path/out.json", None)]


def test_ls_with_info(adl_client_mock):
    fs = ADLGen1FileSystem(adl_client_mock, store_name="dlstore")
    result = list(fs.ls("/path", with_info=True))
    assert result == [
        (
            "/path/to",
            FileInfo(
                file_type=FileType.DIRECTORY, size=0, access_time=None, modify_time=None
            ),
        ),
        (
            "/path/out.json",
            FileInfo(
                file_type=FileType.FILE,
                size=983432,
                access_time=datetime(2020, 6, 14, 9, 33, 26, 477000),
                modify_time=datetime(2020, 9, 9, 7, 36, 14, 996000),
            ),
        ),
    ]


def test_walk_without_info(adl_client_mock):
    fs = ADLGen1FileSystem(adl_client_mock, store_name="dlstore")
    result = list(fs.walk("/path", with_info=False))
    assert result == [
        ("/path/to", None),
        ("/path/out.json", None),
        ("/path/to/file.json", None),
    ]


def test_walk_with_info(adl_client_mock):
    fs = ADLGen1FileSystem(adl_client_mock, store_name="dlstore")
    result = list(fs.walk("/path", with_info=True))
    expected = [
        (
            "/path/to",
            FileInfo(
                file_type=FileType.DIRECTORY, size=0, access_time=None, modify_time=None
            ),
        ),
        (
            "/path/out.json",
            FileInfo(
                file_type=FileType.FILE,
                size=983432,
                access_time=datetime(2020, 6, 14, 9, 33, 26, 477000),
                modify_time=datetime(2020, 9, 9, 7, 36, 14, 996000),
            ),
        ),
        (
            "/path/to/file.json",
            FileInfo(
                file_type=FileType.FILE,
                size=142453,
                access_time=datetime(2020, 9, 8, 22, 21, 39, 143000),
                modify_time=datetime(2020, 9, 8, 22, 40, 56, 564000),
            ),
        ),
    ]
    assert result == expected
