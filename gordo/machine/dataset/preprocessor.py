import logging
import pandas as pd

from typing import Iterable, Dict, Tuple, Union, Type, List
from copy import deepcopy
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from math import floor

logger = logging.getLogger(__name__)


class Preprocessor(metaclass=ABCMeta):
    @abstractmethod
    def reset(self):
        ...

    @abstractmethod
    def prepare_df(self, df: pd.DataFrame) -> pd.DataFrame:
        ...


_types: Dict[str, Type[Preprocessor]] = {}


def preprocessor(preprocessor_type):
    def wrapper(cls):
        if preprocessor_type in _types:
            raise ValueError(
                "Preprocessor with name '%s' has already been added" % preprocessor_type
            )
        _types[preprocessor_type] = cls
        return cls

    return wrapper


def create_preprocessor(preprocessor_type, *args, **kwargs):
    if preprocessor_type not in _types:
        raise ValueError("Can't find a preprocessor with name '%s'" % preprocessor_type)
    return _types[preprocessor_type](*args, **kwargs)


def normalize_preprocessor(value):
    if isinstance(value, dict):
        if "type" not in value:
            raise ValueError("A preprocessor type is empty")
        value = deepcopy(value)
        preprocessor_type = value.pop("type")
        return create_preprocessor(preprocessor_type, **value)
    return value


def gap2str(gap_start: pd.Timestamp, gap_end: pd.Timestamp):
    return "from %s to %s" % (gap_start.isoformat(), gap_end.isoformat())


@preprocessor("mark_gaps")
class MarkGapsPreprocessor(Preprocessor):
    def __init__(
        self,
        gap_size: Union[str, pd.Timedelta],
        mark_value: float,
        fill_nan: bool = False,
    ):
        if isinstance(gap_size, str):
            gap_size = pd.Timedelta(gap_size)
        self.gap_size = gap_size
        self.mark_value = mark_value
        self.fill_nan = fill_nan

    def reset(self):
        pass

    def find_gaps(self, series):
        name = "Time"
        df = pd.concat([series.rename(name), series.diff().rename("Diff")], axis=1)
        filtered_df = df[df["Diff"] > self.gap_size]
        for _, row in filtered_df.iterrows():
            yield row[name], row[name] + row["Diff"]

    def prepare_df(self, df: pd.DataFrame) -> pd.DataFrame:
        index_series = df.index.to_series()
        gaps = list(self.find_gaps(index_series))
        if gaps:
            for from_ts, to_ts in gaps:
                logger.debug("Found gap from %s to %s", from_ts, to_ts)
            for ts, _ in gaps:
                mark_ts = ts + self.gap_size
                for column in df.columns:
                    if self.fill_nan:
                        df[column].fillna(self.mark_value)
                    df.at[mark_ts, column] = self.mark_value
            df = df.sort_index()
        return df
