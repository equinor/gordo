from urllib.parse import quote
from dataclasses import dataclass

from gordo.machine.dataset.file_system import FileSystem
from gordo.machine.dataset.sensor_tag import SensorTag
from .file_type import FileType
from .ncs_file_type import NcsFileType, load_ncs_file_types

from typing import List, Iterable, Tuple, Optional


@dataclass(frozen=True)
class TagLocation:
    tag: SensorTag
    year: int
    exists: bool
    path: Optional[str] = None
    file_type: Optional[FileType] = None


class NcsLookup:
    @classmethod
    def create(
        cls, store: FileSystem, ncs_type_names: Optional[Iterable[str]] = None
    ) -> "NcsLookup":
        ncs_file_types = load_ncs_file_types(ncs_type_names)
        return cls(store, ncs_file_types)

    def __init__(self, store: FileSystem, ncs_file_types: List[NcsFileType]):
        self.store = store
        self.ncs_file_types = ncs_file_types

    @staticmethod
    def quote_tag_name(tag_name: str) -> str:
        return quote(tag_name, safe=" ")

    def tag_dirs_lookup(
        self, base_dir: str, tag_list: List[SensorTag]
    ) -> Iterable[Tuple[SensorTag, Optional[str]]]:
        tags = {}
        for tag in tag_list:
            tag_name = self.quote_tag_name(tag.name)
            tags[tag_name] = tag
        for path, file_info in self.store.ls(base_dir):
            if file_info is not None and file_info.isdir():
                dir_path, file_name = self.store.split(path)
                if file_name in tags:
                    yield tags[file_name], path
                    del tags[file_name]
        for tag in tags.values():
            yield tag, None

    def files_lookup(
        self, tag_dir: str, tag: SensorTag, years: Iterable[int]
    ) -> Iterable[TagLocation]:
        store = self.store
        ncs_file_types = self.ncs_file_types
        tag_name = self.quote_tag_name(tag.name)
        not_existing_years = set(years)
        for year in years:
            found = False
            for ncs_file_type in ncs_file_types:
                for path in ncs_file_type.paths(store, tag_name, year):
                    full_path = store.join(tag_dir, path)
                    if store.exists(full_path):
                        file_type = ncs_file_type.file_type
                        yield TagLocation(
                            tag, year, exists=True, path=full_path, file_type=file_type
                        )
                        found = True
                        break
                if found:
                    not_existing_years.remove(year)
                    break
        for year in not_existing_years:
            yield TagLocation(tag, year, exists=False)
