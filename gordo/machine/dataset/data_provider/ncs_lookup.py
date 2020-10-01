import logging

from urllib.parse import quote
from dataclasses import dataclass

from gordo.machine.dataset.file_system import FileSystem
from gordo.machine.dataset.sensor_tag import SensorTag
from gordo.machine.dataset.exceptions import ConfigException
from .file_type import FileType
from .ncs_file_type import NcsFileType, load_ncs_file_types
from .assets_config import AssetsConfig, PathSpec

from typing import List, Iterable, Tuple, Optional
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


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
        cls,
        store: FileSystem,
        ncs_type_names: Optional[Iterable[str]] = None,
        store_name: Optional[str] = None,
    ) -> "NcsLookup":
        ncs_file_types = load_ncs_file_types(ncs_type_names)
        return cls(store, ncs_file_types, store_name)

    def __init__(
        self,
        store: FileSystem,
        ncs_file_types: List[NcsFileType],
        store_name: Optional[str] = None,
    ):
        self.store = store
        self.ncs_file_types = ncs_file_types
        if store_name is None:
            store_name = store.name
        self.store_name = store_name

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

    def assets_config_tags_lookup(
        self, asset_config: AssetsConfig, tags: List[SensorTag]
    ) -> Iterable[Tuple[SensorTag, str]]:
        store = self.store
        tag_by_assets = OrderedDict()
        for tag in tags:
            if not tag.asset:
                raise ValueError("%s tag has empty asset" % tag.name)
            asset = tag.asset
            if asset not in tag_by_assets:
                tag_by_assets[asset] = list()
            tag_by_assets[asset].append(tag)
        store_name = self.store_name
        asset_path_specs: List[Tuple[PathSpec, List[SensorTag]]] = []
        for asset, asset_tags in tag_by_assets.items():
            path_spec = asset_config.get_path(store_name, asset)
            if path_spec is None:
                raise ValueError(
                    "Unable to find asset '%s' in storage '%s'" % (asset, store_name)
                )
            asset_path_specs.append((path_spec, asset_tags))
        for path_spec, asset_tags in asset_path_specs:
            for tag, tag_dir in self.tag_dirs_lookup(
                path_spec.full_path(store), asset_tags
            ):
                if tag_dir is None:
                    logging.info(
                        "Could not found tag '%s' for asset '%s'"
                        % (tag.name, tag.asset)
                    )
                    continue
                yield tag, tag_dir

    def _thread_pool_lookup_mapper(
        self, tag_dirs: Tuple[SensorTag, str], years: Tuple[int]
    ) -> List[TagLocation]:
        tag, tag_dir = tag_dirs
        tag_locations = []
        for location in self.files_lookup(tag_dir, tag, years):
            tag_locations.append(location)
        return tag_locations

    @staticmethod
    def _years_inf_iterator(years: Tuple[int]) -> Iterable[Tuple[int]]:
        while True:
            yield years

    def lookup(
        self,
        asset_config: AssetsConfig,
        tags: List[SensorTag],
        years: Iterable[int],
        threads_count: int = 1,
    ) -> Iterable[TagLocation]:
        if not threads_count or threads_count < 1:
            raise ConfigException("thread_count should bigger or equal to 1")
        multi_thread = threads_count > 1
        tag_dirs = self.assets_config_tags_lookup(asset_config, tags)
        years_tuple = tuple(years)
        if multi_thread:
            with ThreadPoolExecutor(max_workers=threads_count) as executor:
                result = executor.map(
                    self._thread_pool_lookup_mapper,
                    tag_dirs,
                    self._years_inf_iterator(years_tuple),
                )
                for tag_locations in result:
                    yield from tag_locations
        else:
            for tag, tag_dir in tag_dirs:
                for tag_location in self.files_lookup(tag_dir, tag, years_tuple):
                    yield tag_location
