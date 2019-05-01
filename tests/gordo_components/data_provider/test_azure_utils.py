import logging

import adal
import pytest

from gordo_components.data_provider.azure_utils import get_datalake_token


def test_get_datalake_token_wrong_args():
    with pytest.raises(ValueError):
        get_datalake_token(interactive=False)


def test_get_data_serviceauth_fail(caplog):
    with pytest.raises(adal.adal_error.AdalError), caplog.at_level(logging.CRITICAL):
        get_datalake_token(dl_service_auth_str="TENTANT_UNKNOWN:BOGUS:PASSWORD")
