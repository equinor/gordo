import pytest

from gordo.machine.dataset.exceptions import ConfigException
from gordo.machine.dataset.data_provider.file_type import CsvFileType, ParquetFileType
from gordo.machine.dataset.data_provider.ncs_file_type import (
    NcsCsvFileType,
    NcsParquetFileType,
    load_ncs_file_types,
)


def test_ncs_csv_file_type(mock_file_system):
    ncs_file_type = NcsCsvFileType()
    assert type(ncs_file_type.file_type) is CsvFileType
    paths = ncs_file_type.paths(mock_file_system, "tag1", 2020)
    assert list(paths) == ["tag1_2020.csv"]


def test_ncs_parquet_file_type(mock_file_system):
    ncs_file_type = NcsParquetFileType()
    assert type(ncs_file_type.file_type) is ParquetFileType
    paths = ncs_file_type.paths(mock_file_system, "tag1", 2020)
    assert list(paths) == ["parquet/tag1_2020.parquet"]


def test_load_ncs_file_types():
    ncs_file_types = load_ncs_file_types()
    assert len(ncs_file_types) == 2
    assert type(ncs_file_types[0]) is NcsParquetFileType
    assert type(ncs_file_types[1]) is NcsCsvFileType
    ncs_file_types = load_ncs_file_types(("parquet",))
    assert len(ncs_file_types) == 1
    assert type(ncs_file_types[0]) is NcsParquetFileType


def test_load_ncs_file_types_exception():
    with pytest.raises(ConfigException):
        load_ncs_file_types(["json"])
