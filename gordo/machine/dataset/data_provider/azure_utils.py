# -*- coding: utf-8 -*-
import collections
import re
import os
import logging
from typing import AnyStr

from azure.datalake.store import core, lib
from typing.re import Pattern

logger = logging.getLogger(__name__)


def get_datalake_token(
    interactive: bool = False, dl_service_auth_str: str = None
) -> lib.DataLakeCredential:
    """
    Provides a token for azure datalake, either by parsing a datalake service
    authentication string or using interactive authentication

    Parameters
    ----------
    interactive: bool
        If true then use interactive authentication
    dl_service_auth_str: str
        String on the format tenant:client_id:client_secret

    Returns
    -------
    lib.DataLakeCredential
        A lib.DataLakeCredential which can be used to authenticate towards the datalake
    """
    dl_service_auth_str = (
        os.environ.get("DL_SERVICE_AUTH_STR")
        if str(dl_service_auth_str) == "None"
        else dl_service_auth_str
    )
    if interactive:
        logger.info("Attempting to use interactive azure authentication")
        return lib.auth()
    elif dl_service_auth_str:
        logger.info(f"Attempting to use datalake service authentication")
        dl_service_auth_elems = dl_service_auth_str.split(":")
        tenant = dl_service_auth_elems[0]
        client_id = dl_service_auth_elems[1]
        client_secret = dl_service_auth_elems[2]
        token = lib.auth(
            tenant_id=tenant,
            client_secret=client_secret,
            client_id=client_id,
            resource="https://datalake.azure.net/",
        )
        return token
    else:
        raise ValueError(
            f"Either interactive (value: {interactive}) must be True, "
            f"or dl_service_auth_str (value: {dl_service_auth_str}) "
            "must be set. "
        )


def create_adls_client(
    storename: str, dl_service_auth_str: str = None, interactive: bool = False
) -> core.AzureDLFileSystem:
    """
    Creates an ADLS file system client.

    Parameters
    ----------
    storename: str
        Name of datalake store.
    dl_service_auth_str: str
        Authentication string to use
    interactive: bool
        If true then use interactive authentication


    Returns
    -------
    core.AzureDLFileSystem
        Instance of AzureDLFileSystem, ready to use
    """
    token = get_datalake_token(
        interactive=interactive, dl_service_auth_str=dl_service_auth_str
    )

    adls_file_system_client = core.AzureDLFileSystem(token, store_name=storename)
    return adls_file_system_client


def walk_azure(
    client: core.AzureDLFileSystem,
    base_path: str,
    include_regexp: Pattern[AnyStr] = re.compile(".*"),
    exclude_regexp: Pattern[AnyStr] = re.compile("a^"),
):
    """
    Walks azure datalake `azure_data_store` in a depth-first fashion from
    `base_path`, yielding files which match `include_re` AND not match `exclude_re`

    Notes
    -----
    If `base_path` does not exist then the generator returns nothing, it does not fail.

    """
    logger.info(
        "Starting a azure walk with base_path: {} with include_regexp: {} "
        "and exclude_regexp: {}".format(base_path, include_regexp, exclude_regexp)
    )

    adls_file_system_client = client
    if not adls_file_system_client.exists(base_path):
        return
    fringe = collections.deque(adls_file_system_client.ls(base_path, detail=True))
    while fringe:
        a_path = fringe.pop()
        if a_path["type"] == "DIRECTORY":
            logger.info("Expanding the directory %s" % a_path["name"])
            fringe.extend(adls_file_system_client.ls(a_path["name"], detail=True))
        if a_path["type"] == "FILE":
            file_path = a_path["name"]
            if include_regexp.match(file_path) and not exclude_regexp.match(file_path):
                logger.info("Returning the file_path %s" % file_path)
                yield file_path
            else:
                logger.info(
                    f"Found that the file_path {file_path} does not satisfy regexp "
                    f"requirements"
                )


def is_file(client: core.AzureDLFileSystem, path: str):
    try:
        info = client.info(path)
    except FileNotFoundError:
        return False
    return info["type"] == "FILE"
