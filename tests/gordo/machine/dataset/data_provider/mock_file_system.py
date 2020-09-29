from io import BytesIO
from typing import IO, Iterable, Tuple, Optional
from gordo.machine.dataset.file_system.base import FileSystem, FileInfo


class MockFileSystem(FileSystem):

    @property
    def name(self) -> str:
        return "dlstore"

    def open(self, path: str, mode: str = "r") -> IO:
        return BytesIO()

    def exists(self, path: str) -> bool:
        return False

    def isfile(self, path: str) -> bool:
        return False

    def isdir(self, path: str) -> bool:
        return False

    def info(self, path: str) -> FileInfo:
        raise FileNotFoundError(path)

    def ls(self, path: str, with_info: bool = True) -> Iterable[Tuple[str, Optional[FileInfo]]]:
        return []

    def walk(self, base_path: str, with_info: bool = True) -> Iterable[Tuple[str, Optional[FileInfo]]]:
        return []