import pytest
from mock import patch

from gordo.machine.dataset.file_system.adl1 import ADLGen1FileSystem


@pytest.fixture
def auth_mock():
    with patch("azure.datalake.store.lib.auth") as auth:
        auth.return_value = "123"
        yield auth


@pytest.fixture
def adl_client_mock():
    with patch("azure.datalake.store.core.AzureDLFileSystem") as adl_client:
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
