import os

from typing import Tuple, Optional


def get_env_secret_values(
    dl_service_auth: Optional[str], dl_service_auth_env: str
) -> Tuple[str, str, str]:
    if dl_service_auth is None:
        dl_service_auth = os.environ.get(dl_service_auth_env)
        if not dl_service_auth:
            raise ValueError("Environment variable %s is empty" % dl_service_auth_env)
    data = dl_service_auth.split(":")
    if len(data) != 3:
        raise ValueError("dl_service_auth has %d fields, but 3 is required" % len(data))

    tenant_id, client_id, client_secret = data
    return tenant_id, client_id, client_secret
