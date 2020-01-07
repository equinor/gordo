import logging

import adal
import pytest
from unittest.mock import patch, call

from gordo.machine.dataset.data_provider.azure_utils import get_datalake_token


def test_get_datalake_token_wrong_args():
    with pytest.raises(ValueError):
        get_datalake_token()


@patch("azure.datalake.store.lib.auth")
def test_get_datalake_interactive(mock_stuff):
    get_datalake_token(interactive=True)
    assert mock_stuff.call_args_list == [call()]


@patch("azure.datalake.store.lib.auth")
def test_get_datalake_non_interactive(mock_stuff):
    get_datalake_token(
        interactive=False, dl_service_auth_str="TENTANT_UNKNOWN:BOGUS:PASSWORD"
    )
    assert mock_stuff.call_args_list == [
        call(
            client_id="BOGUS",
            client_secret="PASSWORD",
            resource="https://datalake.azure.net/",
            tenant_id="TENTANT_UNKNOWN",
        )
    ]


def test_get_data_serviceauth_fail(caplog):
    with pytest.raises(adal.adal_error.AdalError), caplog.at_level(logging.CRITICAL):
        get_datalake_token(dl_service_auth_str="TENTANT_UNKNOWN:BOGUS:PASSWORD")
