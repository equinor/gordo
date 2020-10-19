import pandas as pd
import numpy as np

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import IO, Optional


@dataclass
class TimeSeriesColumns:
    """
    Names of columns witch is used in time series datasets
    """

    datetime_column: str
    value_column: str
    status_column: Optional[str] = None

    @property
    def columns(self):
        columns = [self.datetime_column, self.value_column]
        if self.status_column is not None:
            columns.append(self.status_column)
        return columns


class FileType(metaclass=ABCMeta):
    """
    :class:`pandas.DataFrame` reader from the different file types
    """

    file_extension: Optional[str] = None

    @abstractmethod
    def read_df(self, f: IO) -> pd.DataFrame:
        """
        Read `DataFrame` from file object

        Parameters
        ----------
        f : BinaryIO
            File object
        """
        raise NotImplementedError()


class CsvFileType(FileType):

    file_extension: Optional[str] = ".csv"

    def __init__(
        self, header: list, time_series_columns: TimeSeriesColumns, sep: str = ";"
    ):
        """
        Create `DataFrame` reader for CSV files

        Parameters
        ----------
        header: list
            List of all columns in CSV file
        time_series_columns: TimeSeriesColumns
        sep: str
            Delimiter for columns in CSV file
        """
        self.header = header
        self.time_series_columns = time_series_columns
        self.sep = sep

    def read_df(self, f: IO) -> pd.DataFrame:
        datetime_column = self.time_series_columns.datetime_column
        value_column = self.time_series_columns.value_column
        return pd.read_csv(
            f,
            sep=self.sep,
            header=None,
            names=self.header,
            usecols=self.time_series_columns.columns,
            dtype={value_column: np.float32},
            parse_dates=[datetime_column],
            date_parser=lambda col: pd.to_datetime(col, utc=True),
            index_col=datetime_column,
        )


class ParquetFileType(FileType):

    file_extension: str = ".parquet"

    def __init__(self, time_series_columns: TimeSeriesColumns):
        """
        Create `DataFrame` reader for Parquet files

        Parameters
        ----------
        time_series_columns: TimeSeriesColumns
        """
        self.time_series_columns = time_series_columns

    def read_df(self, f: IO) -> pd.DataFrame:
        columns = self.time_series_columns.columns
        datetime_column = self.time_series_columns.datetime_column
        df = pd.read_parquet(f, engine="pyarrow", columns=columns).set_index(
            datetime_column
        )
        df.index = pd.to_datetime(df.index, utc=True)
        return df
