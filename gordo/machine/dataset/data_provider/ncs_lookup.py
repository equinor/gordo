from copy import copy
from urllib.parse import quote
from dataclasses import dataclass

from gordo.machine.dataset.file_system import FileSystem, FileType
from gordo.machine.dataset.sensor_tag import SensorTag
from .file_type import FileType
from .ncs_file_type import NcsFileType, load_ncs_file_types

from typing import List, Iterable, Tuple, Optional, Dict


@dataclass(frozen=True)
class TagLocation:
    tag: SensorTag
    year: int
    exists: bool
    path: Optional[str] = None
    file_type: Optional[FileType] = None


class NcsLookup:

    @classmethod
    def create(cls, store: FileSystem, ncs_type_names: Optional[Iterable[str]] = None) -> "NcsLookup":
        ncs_file_types = load_ncs_file_types(ncs_type_names)
        return cls(store, ncs_file_types)

    def __init__(self, store: FileSystem, ncs_file_types: List[NcsFileType]):
        self.store = store
        self.ncs_file_types = ncs_file_types

    @staticmethod
    def quote_tag_name(tag_name: str) -> str:
        return quote(tag_name, safe=" ")

    def tag_dirs_lookup(self, base_dir: str, tag_list: List[SensorTag]) -> Iterable[Tuple[SensorTag, Optional[str]]]:
        tags = {}
        for tag in tag_list:
            tag_name = self.quote_tag_name(tag.name)
            tags[tag_name] = tag
        for path, file_info in self.store.ls(base_dir):
            if file_info.file_type == FileType.DIRECTORY:
                dir_path, file_name = self.store.split(path)
                if file_name in tags:
                    yield tags[file_name], path
                    del tags[file_name]
        for tag in tags.values():
            yield tag, None

    def files_lookup(self, tag_dir: str, tag: SensorTag, years: Iterable[int]) -> Iterable[TagLocation]:
        fs = self.store
        ncs_file_types = self.ncs_file_types
        tag_name = self.quote_tag_name(tag.name)
        full_paths = {}
        for ncs_ind, ncs_file_type in enumerate(ncs_file_types):
            for year in years:
                for path in ncs_file_type.paths(fs, tag_name, year):
                    full_path = fs.join(tag_dir, path)
                    full_paths[full_path] = (year, ncs_ind)
        locations: Dict[int, List[Optional[TagLocation]]] = dict()
        var_proto = [None] * len(ncs_file_types)
        for year in years:
            locations[year] = copy(var_proto)
        for path, file_info in fs.walk(tag_dir):
            if file_info.file_type == FileType.FILE:
                full_path = fs.join(tag_dir, path)
                if full_path in full_paths:
                    year, ncs_ind = full_paths[full_path]
                    file_type = ncs_file_types[ncs_ind].file_type
                    location = TagLocation(tag, year, exists=True, path=full_path, file_type=file_type)
                    locations[year][ncs_ind] = location
        for year in years:
            found = False
            for location in locations[year]:
                if location is not None:
                    yield location
                    found = True
                    break
            if not found:
                yield TagLocation(tag, year, exists=False)
