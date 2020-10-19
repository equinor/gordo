import logging

from urllib.parse import quote
from dataclasses import dataclass

from gordo.machine.dataset.file_system import FileSystem
from gordo.machine.dataset.sensor_tag import SensorTag
from gordo.machine.dataset.exceptions import ConfigException
from .file_type import FileType
from .ncs_contants import NCS_READER_NAME
from .ncs_file_type import NcsFileType, load_ncs_file_types
from .assets_config import AssetsConfig, PathSpec

from typing import List, Iterable, Tuple, Optional, Dict, Iterator
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Location:
    """
    Represents location of the tag in the data lake
    """

    path: str
    file_type: FileType


@dataclass(frozen=True)
class TagLocations:
    """
    Locations of the tags for each year
    """

    tag: SensorTag
    locations: Optional[Dict[int, Location]] = None

    def available(self) -> bool:
        return self.locations is not None

    def years(self) -> List[int]:
        if self.locations is None:
            return []
        return sorted(self.locations.keys())

    def get_location(self, year: int) -> Optional[Location]:
        if self.locations is None:
            return None
        return self.locations.get(year)

    def __iter__(self) -> Iterator[Tuple[SensorTag, int, Location]]:
        if self.locations is not None:
            locations = self.locations
            for year in self.years():
                yield self.tag, year, locations[year]


class NcsLookup:
    """
    Class which could be used for finding tags data in the data lake storage
    """

    @classmethod
    def create(
        cls,
        storage: FileSystem,
        ncs_type_names: Optional[Iterable[str]] = None,
        storage_name: Optional[str] = None,
    ) -> "NcsLookup":
        ncs_file_types = load_ncs_file_types(ncs_type_names)
        return cls(storage, ncs_file_types, storage_name)

    def __init__(
        self,
        storage: FileSystem,
        ncs_file_types: List[NcsFileType],
        storage_name: Optional[str] = None,
    ):
        self.storage = storage
        self.ncs_file_types = ncs_file_types
        if storage_name is None:
            storage_name = storage.name
        self.storage_name = storage_name

    @staticmethod
    def quote_tag_name(tag_name: str) -> str:
        return quote(tag_name, safe=" ")

    def tag_dirs_lookup(
        self, base_dir: str, tag_list: List[SensorTag]
    ) -> Iterable[Tuple[SensorTag, Optional[str]]]:
        """
        Takes list of tags and find directories related to them in the data lake storage

        Parameters
        ----------
        base_dir: str
            Base directory where all of the tags are placed
        tag_list: List[SensorTag]
            List of the tags

        Returns
        -------

        """
        tags = {}
        for tag in tag_list:
            tag_name = self.quote_tag_name(tag.name)
            tags[tag_name] = tag
        for path, file_info in self.storage.ls(base_dir):
            if file_info is not None and file_info.isdir():
                dir_path, file_name = self.storage.split(path)
                if file_name in tags:
                    yield tags[file_name], path
                    del tags[file_name]
        for tag in tags.values():
            yield tag, None

    def files_lookup(
        self, tag_dir: str, tag: SensorTag, years: Iterable[int]
    ) -> TagLocations:
        """
        Finds files (possible parquet or CSV) in tag directory in the data lake storage

        Parameters
        ----------
        tag_dir: str
        tag: SensorTag
        years: Iterable[int]
            List of the years for finding files

        Returns
        -------
        TagLocations

        """
        storage = self.storage
        ncs_file_types = self.ncs_file_types
        tag_name = self.quote_tag_name(tag.name)
        not_existing_years = set(years)
        locations = {}
        for year in years:
            found = False
            for ncs_file_type in ncs_file_types:
                for path in ncs_file_type.paths(storage, tag_name, year):
                    full_path = storage.join(tag_dir, path)
                    if storage.exists(full_path):
                        file_type = ncs_file_type.file_type
                        locations[year] = Location(full_path, file_type)
                        found = True
                        break
                if found:
                    not_existing_years.remove(year)
                    break
        return TagLocations(tag, locations if locations else None)

    def assets_config_tags_lookup(
        self,
        asset_config: AssetsConfig,
        tags: List[SensorTag],
        base_dir: Optional[str] = None,
    ) -> Iterable[Tuple[SensorTag, Optional[str]]]:
        """
        Takes assets paths from ``AssetsConfig`` and find tag paths in the data lake storage

        Parameters
        ----------
        asset_config: AssetsConfig
        tags: List[SensorTag]
        base_dir: Optional[str]

        Returns
        -------

        """
        storage = self.storage
        asset_path_specs: List[Tuple[PathSpec, List[SensorTag]]] = []
        if not base_dir:
            tag_by_assets: Dict[str, List[SensorTag]] = OrderedDict()
            for tag in tags:
                if not tag.asset:
                    raise ValueError("%s tag has empty asset" % tag.name)
                asset = tag.asset
                if asset not in tag_by_assets:
                    tag_by_assets[asset] = list()
                tag_by_assets[asset].append(tag)
            storage_name = self.storage_name
            for asset, asset_tags in tag_by_assets.items():
                path_spec = asset_config.get_path(storage_name, asset)
                if path_spec is None:
                    raise ValueError(
                        "Unable to find asset '%s' in storage '%s'"
                        % (asset, storage_name)
                    )
                if path_spec.reader != NCS_READER_NAME:
                    raise ValueError(
                        "Assets reader name should be equal '%s' and not '%s'"
                        % (NCS_READER_NAME, path_spec.reader)
                    )
                asset_path_specs.append((path_spec, asset_tags))
        else:
            path_spec = PathSpec(NCS_READER_NAME, base_dir, "")
            asset_path_specs.append((path_spec, tags))
        for path_spec, asset_tags in asset_path_specs:
            for tag, tag_dir in self.tag_dirs_lookup(
                path_spec.full_path(storage), asset_tags
            ):
                yield tag, tag_dir

    def _thread_pool_lookup_mapper(
        self, tag_dirs: Tuple[SensorTag, Optional[str]], years: List[int]
    ) -> TagLocations:
        tag, tag_dir = tag_dirs
        if tag_dir is not None:
            return self.files_lookup(tag_dir, tag, years)
        else:
            return TagLocations(tag, None)

    @staticmethod
    def _years_inf_iterator(years: Iterable[int]) -> Iterable[Iterable[int]]:
        while True:
            yield years

    def lookup(
        self,
        asset_config: AssetsConfig,
        tags: List[SensorTag],
        years: Iterable[int],
        threads_count: int = 1,
        base_dir: Optional[str] = None,
    ) -> Iterable[TagLocations]:
        """
        Takes assets paths from ``AssetsConfig`` and find tags files paths in the data lake storage

        Parameters
        ----------
        asset_config: AssetsConfig
        tags: List[SensorTag]
        years: Iterable[int]
        threads_count: int
            Number of threads for internal `ThreadPool`. Do not uses thread pool if 1
        base_dir: Optional[str]

        Returns
        -------

        """
        if not threads_count or threads_count < 1:
            raise ConfigException("thread_count should bigger or equal to 1")
        multi_thread = threads_count > 1
        tag_dirs = self.assets_config_tags_lookup(asset_config, tags, base_dir=base_dir)
        years_tuple = tuple(years)
        if multi_thread:
            with ThreadPoolExecutor(max_workers=threads_count) as executor:
                result = executor.map(
                    self._thread_pool_lookup_mapper,
                    tag_dirs,
                    self._years_inf_iterator(years_tuple),
                )
                for tag_locations in result:
                    yield tag_locations
        else:
            for tag, tag_dir in tag_dirs:
                if tag_dir is not None:
                    yield self.files_lookup(tag_dir, tag, years_tuple)
                else:
                    yield TagLocations(tag, None)
