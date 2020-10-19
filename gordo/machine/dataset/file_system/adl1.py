import logging

from datetime import datetime
from io import TextIOWrapper
from azure.datalake.store import core, lib
from typing import Optional, Iterable, IO, Tuple

from .base import FileSystem, FileInfo, FileType
from .utils import get_env_secret_values

logger = logging.getLogger(__name__)


def time_from_info(info: dict, time_key: str) -> Optional[datetime]:
    if time_key in info:
        unix_timestamp = info[time_key]
        return datetime.utcfromtimestamp(unix_timestamp / 1000)
    else:
        return None


class ADLGen1FileSystem(FileSystem):
    @classmethod
    def create_from_env(
        cls,
        store_name: str,
        dl_service_auth: Optional[str] = None,
        interactive: bool = False,
        dl_service_auth_env: str = "DL_SERVICE_AUTH_STR",
    ) -> "ADLGen1FileSystem":
        """
        Creates ADL Gen1 file system client.

        Parameters
        ----------
        store_name: str
            Name of datalake store.
        dl_service_auth: str
            Authentication string to use. `:` separated values of: tenant_id, client_id, client_secret.
        interactive: bool
            If true then use interactive authentication
        dl_service_auth_env: str
            Environment variable which contains dl_service_auth. DL_SERVICE_AUTH_STR by default

        Returns
        -------
        ADLGen1FileSystem
        """

        if interactive:
            logger.info("Attempting to use interactive azure authentication")
            token = lib.auth()
        else:
            logger.info(f"Attempting to use datalake service authentication")
            tenant_id, client_id, client_secret = get_env_secret_values(
                dl_service_auth, dl_service_auth_env
            )
            token = lib.auth(
                tenant_id=tenant_id,
                client_id=client_id,
                client_secret=client_secret,
                resource="https://datalake.azure.net/",
            )

        adl_client = core.AzureDLFileSystem(token, store_name=store_name)
        return cls(adl_client, store_name)

    def __init__(self, adl_client: core.AzureDLFileSystem, store_name: str):
        self.adl_client = adl_client
        self.store_name = store_name

    @property
    def name(self):
        return self.store_name

    def open(self, path: str, mode: str = "r") -> IO:
        for m in mode:
            if m not in "rb":
                raise ValueError("Unsupported file open mode '%s'" % m)
        wrap_as_text = False
        if "b" not in mode:
            wrap_as_text = True
            mode += "b"
        fd = self.adl_client.open(path, mode=mode)
        return TextIOWrapper(fd) if wrap_as_text else fd

    def exists(self, path: str) -> bool:
        return self.adl_client.exists(path)

    def isfile(self, path: str) -> bool:
        adl_client = self.adl_client
        if adl_client.exists(path):
            info = adl_client.info(path)
            return info["type"] == "FILE"
        else:
            return False

    def isdir(self, path: str) -> bool:
        adl_client = self.adl_client
        if adl_client.exists(path):
            info = adl_client.info(path)
            return info["type"] == "DIRECTORY"
        else:
            return False

    def info(self, path: str) -> FileInfo:
        info = self.adl_client.info(path)
        return self.prepare_info(info)

    @staticmethod
    def prepare_info(info: dict) -> FileInfo:
        if info["type"] == "FILE":
            file_type = FileType.FILE
        elif info["type"] == "DIRECTORY":
            file_type = FileType.DIRECTORY
        else:
            raise ValueError("Unsupported file type '%s'" % info["type"])
        return FileInfo(
            file_type,
            info.get("length", 0),
            access_time=time_from_info(info, "accessTime"),
            modify_time=time_from_info(info, "modificationTime"),
        )

    def ls(
        self, path: str, with_info: bool = True
    ) -> Iterable[Tuple[str, Optional[FileInfo]]]:
        for info in self.adl_client.ls(path, detail=with_info):
            file_path = info["name"] if with_info else info
            file_info = self.prepare_info(info) if with_info else None
            yield file_path, file_info

    def walk(
        self, base_path: str, with_info: bool = True
    ) -> Iterable[Tuple[str, Optional[FileInfo]]]:
        child_directories = []

        for info in self.adl_client.ls(base_path, detail=True):
            file_path = info["name"]
            file_info = self.prepare_info(info) if with_info else None
            if info["type"] == "DIRECTORY":
                child_directories.append(file_path)
            yield file_path, file_info

        for child_directory in child_directories:
            for tup in self.walk(child_directory, with_info=with_info):
                yield tup
