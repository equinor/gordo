import posixpath

from typing import Optional, Iterable, IO, Tuple
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from datetime import datetime

from enum import Enum


class FileType(Enum):
    DIRECTORY = 1
    FILE = 2


@dataclass(frozen=True)
class FileInfo:
    file_type: FileType
    size: int
    access_time: Optional[datetime] = None
    modify_time: Optional[datetime] = None
    create_time: Optional[datetime] = None

    def isfile(self) -> bool:
        return self.file_type == FileType.FILE

    def isdir(self) -> bool:
        return self.file_type == FileType.DIRECTORY


def default_join(*p) -> str:
    if p and not p[-1]:
        p = p[:-1]
    if not p:
        return ""
    return posixpath.join(*p)


class FileSystem(metaclass=ABCMeta):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def open(self, path: str, mode: str = "r") -> IO:
        """
        Open a file

        Parameters
        ----------
        path
            Path to the file
        mode
            Required support at least of `r`,`b` modes
        Returns
        -------
        IOBase
        """
        ...

    @abstractmethod
    def exists(self, path: str) -> bool:
        ...

    @abstractmethod
    def isfile(self, path: str) -> bool:
        ...

    @abstractmethod
    def isdir(self, path: str) -> bool:
        ...

    @abstractmethod
    def info(self, path: str) -> FileInfo:
        ...

    @abstractmethod
    def ls(
        self, path: str, with_info: bool = True
    ) -> Iterable[Tuple[str, Optional[FileInfo]]]:
        ...

    @abstractmethod
    def walk(
        self, base_path: str, with_info: bool = True
    ) -> Iterable[Tuple[str, Optional[FileInfo]]]:
        ...

    def join(self, *p) -> str:
        return default_join(*p)

    def split(self, p) -> Tuple[str, str]:
        return posixpath.split(p)
