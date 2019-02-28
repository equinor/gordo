import unittest
import adal

from gordo_components.data_provider.azure_utils import get_datalake_token


class AzureUtilsTestCase(unittest.TestCase):
    def test_get_datalake_token_wrong_args(self):
        with self.assertRaises(ValueError):
            get_datalake_token(interactive=False)

    def test_get_data_serviceauth_fail(self):
        with self.assertRaises(adal.adal_error.AdalError):
            get_datalake_token(dl_service_auth_str="TENTANT_UNKNOWN:BOGUS:PASSWORD")
