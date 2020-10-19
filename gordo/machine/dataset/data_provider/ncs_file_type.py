from abc import ABCMeta, abstractmethod

from gordo.machine.dataset.file_system import FileSystem
from .file_type import FileType, ParquetFileType, CsvFileType, TimeSeriesColumns

from typing import Iterable, Optional, List

from ..exceptions import ConfigException

time_series_columns = TimeSeriesColumns("Time", "Value", "Status")


class NcsFileType(metaclass=ABCMeta):
    """
    Represents logic about finding files of one particular type for ``NcsLookup``
    """

    @property
    @abstractmethod
    def file_type(self) -> FileType:

        ...

    @abstractmethod
    def paths(self, fs: FileSystem, tag_name: str, year: int) -> Iterable[str]:
        """
        Possible file paths for this file type. These paths should be relational to the tag directory

        Parameters
        ----------
        fs: FileSystem
        tag_name: str
        year: int

        Returns
        -------
        Iterable[str]

        """
        ...


class NcsParquetFileType(NcsFileType):
    """
    NCS parquet files finder
    """

    def __init__(self):
        self._file_type = ParquetFileType(time_series_columns)

    @property
    def file_type(self) -> FileType:
        return self._file_type

    def paths(self, fs: FileSystem, tag_name: str, year: int) -> Iterable[str]:
        file_extension = self._file_type.file_extension
        return (fs.join("parquet", f"{tag_name}_{year}{file_extension}"),)


class NcsCsvFileType(NcsFileType):
    """
    NCS CSV files finder
    """

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
    """
    Returns list of ``NcsFileType`` instances from names of those types

    Parameters
    ----------
    type_names: Optional[Iterable[str]]
        List of ``NcsFileType`` names. Only supporting `parquet` and `csv` names values

    Returns
    -------
    List[NcsFileType]

    """
    if type_names is None:
        type_names = DEFAULT_TYPE_NAMES
    result = []
    for type_name in type_names:
        if type_name not in ncs_file_types:
            raise ConfigException("Can not find file type '%s'" % type_name)
        result.append(ncs_file_types[type_name]())
    return result
