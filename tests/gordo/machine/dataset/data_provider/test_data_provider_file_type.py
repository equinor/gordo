import os
import pandas as pd

from pytest import fixture

from gordo.machine.dataset.data_provider.file_type import (
    CsvFileType,
    ParquetFileType,
    TimeSeriesColumns,
)


@fixture
def data_dir():
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "datalake")


def compare_dtype_names(dtypes, column_types):
    for column, type_name in column_types:
        try:
            dtype = dtypes[column]
        except KeyError:
            raise AssertionError(
                "Can not find '%s' column in pandas.DataFrame" % column
            )
        assert dtype.name == type_name, "Wrong dtype for column '%s'" % column


def test_csv_file_type(data_dir):
    csv_file = os.path.join(data_dir, "TRC-322", "TRC-322_2000.csv")
    time_series_columns = TimeSeriesColumns("Time", "Value", "Status")
    header = ["Sensor", "Value", "Time", "Status"]
    file_type = CsvFileType(header, time_series_columns)
    with open(csv_file, "rb") as f:
        df = file_type.read_df(f)
        assert len(df) == 10
        compare_dtype_names(df.dtypes, (("Value", "float32"), ("Status", "int64")))
        assert isinstance(df.index, pd.DatetimeIndex)


def test_parquet_file_type(data_dir):
    parquet_file = os.path.join(data_dir, "TRC-323", "parquet", "TRC-323_2001.parquet")
    time_series_columns = TimeSeriesColumns("Time", "Value", "Status")
    file_type = ParquetFileType(time_series_columns)
    with open(parquet_file, "rb") as f:
        df = file_type.read_df(f)
        assert len(df) == 20
        compare_dtype_names(df.dtypes, (("Value", "float64"), ("Status", "int32")))
        assert isinstance(df.index, pd.DatetimeIndex)
