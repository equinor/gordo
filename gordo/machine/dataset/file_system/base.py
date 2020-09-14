import posixpath

from typing import Optional, Iterable
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from datetime import datetime

from io import IOBase
from enum import Enum


class FileType(Enum):
    DIRECTORY = 1
    FILE = 2


@dataclass(frozen=True)
class FileInfo:
    file_type: FileType
    size: int
    access_time: Optional[datetime]
    modify_time: Optional[datetime]


class FileSystem(metaclass=ABCMeta):
    @abstractmethod
    def open(self, path: str, mode: str = "r") -> IOBase:
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
    def info(self, path: str) -> Optional[FileInfo]:
        ...

    @abstractmethod
    def walk(self, base_path: str) -> Iterable[str]:
        ...

    def join(self, *p):
        posixpath.join(*p)
