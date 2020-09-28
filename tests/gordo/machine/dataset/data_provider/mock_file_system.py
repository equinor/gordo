import posixpath

from io import BytesIO
from typing import IO, Iterable, Optional, List
from gordo.machine.dataset.file_system.base import FileSystem, FileInfo, FileType, Tuple


class MockFileSystem(FileSystem):

    def __init__(self, tree: Tuple = (), name: str = "dlstore"):
        self.tree = tree
        self._name = name
        self.validate_tree(self.tree)

    def validate_tree(self, tree):
        for file_name, child_tree in tree:
            if type(file_name) is not str:
                raise ValueError("Validation error: "+str(file_name))
            if child_tree is None:
                pass
            elif type(child_tree) is tuple:
                if len(child_tree) != 2:
                    raise ValueError("Validation error: "+str(child_tree))
            else:
                raise ValueError("Validation error: "+str(child_tree))
            if child_tree is not None:
                self.validate_tree(child_tree)

    @staticmethod
    def split_path(path: str) -> List[str]:
        result = path.split(posixpath.sep)
        return result[1:] if not result[0] else result

    def find_path(self, path: List[str], curr_tree: Optional[Tuple] = None) -> Optional[Tuple]:
        if curr_tree is None:
            curr_tree = self.tree
        if not path:
            return curr_tree
        file_name = path[0]
        found = False
        new_tree = None
        for name, tree in curr_tree:
            if name == file_name:
                new_tree = tree
                found = True
                break
        if not found:
            raise FileNotFoundError()
        if new_tree is None:
            return None
        return self.find_path(path[1:], new_tree)

    @property
    def name(self) -> str:
        return self._name

    def open(self, path: str, mode: str = "r") -> IO:
        return BytesIO(b"")

    def exists(self, path: str) -> bool:
        try:
            self.info(path)
            return True
        except FileNotFoundError:
            return False

    def isfile(self, path: str) -> bool:
        return self.info(path).file_type == FileType.FILE

    def isdir(self, path: str) -> bool:
        return self.info(path).file_type == FileType.DIRECTORY

    def info(self, path: str) -> FileInfo:
        tree = self.find_path(self.split_path(path))
        info = FileInfo(FileType.FILE if tree is None else FileType.DIRECTORY, 0)
        return info

    def ls(self, path: str, with_info: bool = True) -> Iterable[Tuple[str, Optional[FileInfo]]]:
        tree = self.find_path(self.split_path(path))
        if tree is None:
            raise FileNotFoundError()
        for file_name in tree:
            file_path = self.join(path, file_name)
            yield file_path, self.info(file_path) if with_info else None

    def walk(self, base_path: str, with_info: bool = True) -> Iterable[Tuple[str, Optional[FileInfo]]]:
        child_directories = []
        for path, file_info in self.ls(base_path, with_info=with_info):
            if file_info.file_type == FileType.DIRECTORY:
                child_directories.append(path)
            yield path, file_info

        for child_directory in child_directories:
            for tup in self.walk(child_directory, with_info=with_info):
                yield tup
