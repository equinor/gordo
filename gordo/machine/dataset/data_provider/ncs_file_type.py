from abc import ABCMeta, abstractmethod

from gordo.machine.dataset.file_system import FileSystem
from .file_type import FileType, ParquetFileType, CsvFileType, TimeSeriesColumns

from typing import Iterable, Optional, List

time_series_columns = TimeSeriesColumns("Time", "Value", "Status")


class NcsFileType(metaclass=ABCMeta):
    @property
    @abstractmethod
    def file_type(self) -> FileType:
        ...

    @abstractmethod
    def paths(self, fs: FileSystem, tag_name: str, year: int) -> Iterable[str]:
        ...


class NcsParquetFileType(NcsFileType):
    def __init__(self):
        self._file_type = ParquetFileType(time_series_columns)

    @property
    def file_type(self) -> FileType:
        return self._file_type

    def paths(self, fs: FileSystem, tag_name: str, year: int) -> Iterable[str]:
        file_extension = self._file_type.file_extension
        return (fs.join("parquet", f"{tag_name}_{year}{file_extension}"),)


class NcsCsvFileType(NcsFileType):
    def __init__(self):
        header = ["Sensor", "Value", "Time", "Status"]
        self._file_type = CsvFileType(header, time_series_columns)

    @property
    def file_type(self) -> FileType:
        return self._file_type

    def paths(self, fs: FileSystem, tag_name: str, year: int) -> Iterable[str]:
        file_extension = self._file_type.file_extension
        return (f"{tag_name}_{year}{file_extension}",)


ncs_file_types = {
    "parquet": NcsParquetFileType,
    "csv": NcsCsvFileType,
}

DEFAULT_TYPE_NAMES = ("parquet", "csv")


def load_ncs_file_types(
    type_names: Optional[Iterable[str]] = None,
) -> List[NcsFileType]:
    if type_names is None:
        type_names = DEFAULT_TYPE_NAMES
    return [ncs_file_types[type_name]() for type_name in type_names]
